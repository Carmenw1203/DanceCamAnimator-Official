import multiprocessing
import os
import sys
import copy
sys.path.append("..")
sys.path.append(".")
import pickle
from functools import partial
from pathlib import Path
import json
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
from my_utils.torch_glm import torch_glm_translate, torch_glm_rotate
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import reduce
from my_utils.vis import MotionCameraRenderSave
from dataset.dckm_dataset import DCKMDataset
from model.EDC_model import MusicDanceCameraKeyframeDecoder
from model.adan import Adan
from dataset.preprocess import increment_path
import math

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
        
class MusicDance2Keyframe:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer_pose= None,
        EMA_tag=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        w_positive_loss=3.0,
        w_negative_loss=0.5,
        loss_type="bce",
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_aist_feats = feature_type == "aist"

        self.motion_repr_dim = motion_repr_dim = 60 * 3 # 60 joints, keypoints3d

        feature_dim = 35 if use_aist_feats else 4800
        
        history_seconds = 2
        inference_seconds = 2 
        FPS = 30
        self.history_len = history_len = history_seconds * FPS
        self.inference_len = inference_len = inference_seconds * FPS

        self.w_positive_loss = w_positive_loss
        self.w_negative_loss = w_negative_loss
        self.ema = EMA(0.9999)

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer_pose = checkpoint["normalizer_pose"]
        # model (to do)
        model = MusicDanceCameraKeyframeDecoder(
            nfeats=1,
            history_len=history_len,
            inference_len=inference_len,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            m_cond_feature_dim=feature_dim,
            p_cond_dim=motion_repr_dim,
            activation=F.gelu,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)

        self.master_model = copy.deepcopy(self.model)

        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA_tag else "model_state_dict"],
                    num_processes,
                )
            )

        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([self.w_negative_loss, self.w_positive_loss]),reduction="none").to(self.accelerator.device)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)
    
    def load_datasets(self, opt):
        
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = DCKMDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
                history_len = self.history_len,
                inference_len = self.inference_len,
                stride_len = 15,
                feature_type = opt.feature_type
            )
            test_dataset = DCKMDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer_pose= train_dataset.normalizer_pose,
                force_reload=opt.force_reload,
                history_len = self.history_len,
                inference_len = self.inference_len,
                stride_len = self.inference_len,
                feature_type = opt.feature_type
            )

            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        return train_dataset, test_dataset

    def train_loop(self, opt):
        # load datasets
        train_dataset, test_dataset = self.load_datasets(opt)
        # set normalizer
        self.normalizer_pose = test_dataset.normalizer_pose

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=test_dataset.subsequence_end_index[-1],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )

        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()

        for epoch in tqdm(range(1, opt.epochs + 1)):
            avg_loss = 0
            total_element = 0
            total_correct = 0
            pos_total_element = 0
            pos_total_correct = 0
            neg_total_element = 0
            neg_total_correct = 0
            ones_cnt = 0
            # train
            self.train()

            for step, (x_camera_kf, x_padding_mask, pose_cond, music_cond, x_wavpath) in enumerate(
                load_loop(train_data_loader)
            ):
                x_camera_kf_res = self.model(
                    x_camera_kf, x_padding_mask, pose_cond, music_cond
                )
                x_camera_kf_inference = x_camera_kf[:,self.history_len:] # b x self.inference_len x 1
                x_camera_kf_res_inference = x_camera_kf_res[:,self.history_len:]#b x self.inference_len x 2
                x_camera_kf_res_inference_transpose = x_camera_kf_res_inference.transpose(1,2)#b x self.inference_len x 2 -> b x 2 x self.inference_len

                total_loss = self.loss_fn(x_camera_kf_res_inference_transpose, x_camera_kf_inference[:,:,0])# b x self.inference_len
                total_loss = total_loss * x_padding_mask[:,self.history_len:,0].float()

                self.optim.zero_grad()
                self.accelerator.backward(total_loss.mean())

                self.optim.step()

                # ema update and train loss update only on main
                
                if self.accelerator.is_main_process:
                    avg_loss += total_loss.mean().detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.ema.update_model_average(
                            self.master_model, self.model
                        )
                    #log
                    correct = (x_camera_kf_res_inference.argmax(dim=-1, keepdims=True).eq(x_camera_kf_inference)*x_padding_mask[:,self.history_len:]).sum().item()
                    total_element += x_padding_mask[:,self.history_len:].sum()
                    total_correct += correct
                    
                    pos_idx = torch.nonzero(x_camera_kf_inference*x_padding_mask[:,self.history_len:] == 1) 
                    pos_correct = ((x_camera_kf_res_inference.argmax(dim=-1, keepdims=True)+x_camera_kf_inference)*x_padding_mask[:,self.history_len:] == 2).sum()
                    pos_total_correct += pos_correct
                    pos_total_element += len(pos_idx)

                    neg_idx = torch.nonzero((x_camera_kf_inference+1)*x_padding_mask[:,self.history_len:] == 1)
                    neg_correct = ((x_camera_kf_res_inference.argmax(dim=-1, keepdims=True)+x_camera_kf_inference+2)*x_padding_mask[:,self.history_len:] == 2).sum()
                    neg_total_correct += neg_correct
                    neg_total_element += len(neg_idx)
                    # pdb.set_trace()

                    ones = (x_camera_kf_res_inference.argmax(dim=-1, keepdims=True)*x_padding_mask[:,self.history_len:]).sum().item()
                    ones_cnt += ones
                    
                    assert total_element == pos_total_element + neg_total_element
                    log_dict = {
                        "Train Loss": avg_loss/ len(train_data_loader),
                        "Avg acc": total_correct / total_element,
                        "Avg pos acc": pos_total_correct / pos_total_element,
                        "Avg neg acc": neg_total_correct / neg_total_element,
                        "ones_predicted": ones_cnt / total_element
                    }
                    wandb.log(log_dict)
            # Save model
            if (epoch % opt.save_interval) == 0:
                
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # print('A:',torch.cuda.memory_allocated(),torch.cuda.max_memory_cached())
                    # log
                    ckpt = {
                        "ema_state_dict": self.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer_pose": self.normalizer_pose,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # generate a sample
                    render_count = -1
                    sample_batch_size = test_dataset.subsequence_end_index[render_count]
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x_camera_kf, x_padding_mask, pose_cond, music_cond, x_wavpath) = next(iter(test_data_loader))
                    pose_cond = pose_cond.to(self.accelerator.device)
                    music_cond = music_cond.to(self.accelerator.device)
                    self.render_sample(
                        sample_batch_size,
                        pose_cond,
                        music_cond,
                        x_camera_kf,
                        x_padding_mask,
                        x_wavpath,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name)
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
            
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(self, sample_batch_size, pose_cond, music_cond, x_camera_kf, x_padding_mask, x_wavpath,
                         epoch, out_dir, is_train = True):
        
        Path(os.path.join(out_dir, str(epoch))).mkdir(parents=True, exist_ok=True)
        pose_cond = pose_cond.to(self.accelerator.device)#gt
        music_cond = music_cond.to(self.accelerator.device)#gt
        x_camera_kf = x_camera_kf.to(self.accelerator.device)#gt
        x_padding_mask = x_padding_mask.to(self.accelerator.device)#gt

        sample_camera_kmask = None
        sample_camera_kmask_cond = None
        i_sample = 0
        pbar = tqdm(total=sample_batch_size)
        # synthesize all results following an auto-regressive scheme, stitch results within the same audio. this need the data be ordered, so it's better to not shuffle the data for this function
        while(i_sample < sample_batch_size):
            tmp_imask_len = int(torch.sum(x_padding_mask[i_sample,self.history_len:]))# denotes how many frames should be synthesized
            if sample_camera_kmask == None:#means here comes a subsequence of a new audio and the history is empty *0
                if is_train:
                    sample_result = self.model.module(x_camera_kf[i_sample:i_sample+1]*0, x_padding_mask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                else:
                    # print(x_camera[i_sample:i_sample+1:])
                    # print(0)
                    sample_result = self.model(x_camera_kf[i_sample:i_sample+1]*0, x_padding_mask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                # print(sample_result[:,self.history_len:self.history_len+1,:])/
            else:# means the previous sample is the history of this sample, so we can use the record sample_camera_kmask_cond from previous sample as history
                if is_train: 
                    sample_result = self.model.module(sample_camera_kmask_cond[0:1], x_padding_mask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                else:
                    sample_result = self.model(sample_camera_kmask_cond[0:1], x_padding_mask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
            
            sample_camera_kmask_cond =  torch.zeros_like(x_camera_kf[i_sample:i_sample+1])
            
            # record history
            sample_camera_kmask_cond[:,:self.history_len,:] = sample_result[:,tmp_imask_len:self.history_len+tmp_imask_len,:].argmax(dim=-1, keepdims=True).float().detach()

            sample_result = sample_result.detach().cpu()

            # synthesize results
            if sample_camera_kmask == None: # means start of a piece of test audio
                sample_camera_kmask = sample_result[0:1,self.history_len:self.history_len+tmp_imask_len,:].argmax(dim=-1, keepdims=True)
            else: # means not a start of a piece of test audio, then stitch
                sample_camera_kmask = torch.cat([sample_camera_kmask, sample_result[0:1,self.history_len:self.history_len+tmp_imask_len,:].argmax(dim=-1, keepdims=True)], dim=1)
            # save the camera keyframes
            if i_sample + 1 == sample_batch_size: # this is the last sub-sequence for this batch
                self.save_camera_keyframe(sample_camera_kmask, epoch, out_dir, x_wavpath[i_sample])
                # reset
                sample_camera_kmask = None 
                sample_camera_kmask_cond = None
                i_sample += 1
            elif not x_wavpath[i_sample + 1] == x_wavpath[i_sample]:#here is an end for a piece of data
                self.save_camera_keyframe(sample_camera_kmask, epoch, out_dir, x_wavpath[i_sample])
                # reset
                sample_camera_kmask = None 
                sample_camera_kmask_cond = None
                i_sample += 1
            else:#current audio not end
                i_sample += 1
            pbar.update(1)
            
    def save_camera_keyframe(self, c_kf, epoch, render_out, wav_path):
        # camera_keyframe_name
        norm_wav_path = os.path.normpath(wav_path)# path is like "DCM++/Test/Audio/a3_1_C.wav"
        pathparts = norm_wav_path.split(os.sep)
        camera_name = 'ck' + pathparts[-1][1:-6] + '.json'
        with open(os.path.join(render_out,str(epoch),camera_name), 'w') as ckf:
            json.dump({
                "KeyframeMask":c_kf[0,:,0].numpy().tolist(),
                "KeyframePos":torch.nonzero(c_kf == 1)[:,1].numpy().tolist()
            }, ckf)
