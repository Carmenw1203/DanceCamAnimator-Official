# Editable dance camera synthesis with keyframes
# filming dance like an animator using keyframes
# stage2 given music motion and keyframe temporal position of camera movement, inference camera keyframe motions
# stage3 interpolate non-keyframe movement between keyframe motions
import multiprocessing
import os
import sys
import copy
sys.path.append("..")
sys.path.append(".")
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from my_utils.torch_glm import torch_glm_translate, torch_glm_rotate
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import reduce
from my_utils.vis import MotionCameraRenderSave
from dataset.dcmpp_dataset import DCMPPDataset
from model.EDC_model import EditableDanceCameraDecoder_ForVelocity
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



class EditableDanceCameraVelocity:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer_pose= None,
        normalizer_camera_dis = None,
        normalizer_camera_pos = None,
        normalizer_camera_rot = None,
        normalizer_camera_fov = None,
        normalizer_camera_eye = None,
        EMA_tag=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        camera_format='polar', # only works for polar representation
        w_loss=2,
        w_v_loss=5,
        w_a_loss=5,
        w_in_ba_loss=0.0015,
        w_out_ba_loss=0,
        loss_type="l2",
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_aist_feats = feature_type == "aist"

        self.motion_repr_dim = motion_repr_dim = 60 * 3 # 60 joints, keypoints3d
        self.camera_format = camera_format
        if camera_format == 'polar':
            self.camera_repr_dim = camera_repr_dim = 1+ 3 * 2 + 1 # dis, pos, rot, fov
        elif camera_format == 'centric':
            self.camera_repr_dim = camera_repr_dim = 3 + 1 + 3 # rot, fov, eye

        feature_dim = 35 if use_aist_feats else 4800
        
        history_seconds = 2
        inference_seconds = 2 
        FPS = 30
        self.history_len = history_len = history_seconds * FPS
        self.inference_len = inference_len = inference_seconds * FPS

        self.w_loss = w_loss
        self.w_v_loss = w_v_loss
        self.w_a_loss = w_a_loss
        self.w_in_ba_loss = w_in_ba_loss
        self.w_out_ba_loss = w_out_ba_loss
        self.ema = EMA(0.9999)

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer_pose = checkpoint["normalizer_pose"]
            self.normalizer_camera_dis = checkpoint["normalizer_camera_dis"]
            self.normalizer_camera_pos = checkpoint["normalizer_camera_pos"]
            self.normalizer_camera_rot = checkpoint["normalizer_camera_rot"]
            self.normalizer_camera_fov = checkpoint["normalizer_camera_fov"]
            # if camera_format == 'centric':
            #     self.normalizer_camera_eye = checkpoint["normalizer_camera_eye"]
            # else:
            #     self.normalizer_camera_eye = None
            self.normalizer_camera_eye = checkpoint["normalizer_camera_eye"]
        
        model = EditableDanceCameraDecoder_ForVelocity(
            nfeats=camera_repr_dim,
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

        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def calculate_losses(self, camera_res, camera_gt, camera_imask, b_mask, p_cond):
        camera_imask_sum = reduce(camera_imask, "b t c -> b c", "sum")
        # full reconstruction loss
        loss = self.loss_fn(camera_res, camera_gt, reduction="none")
        
        loss = loss*camera_imask
        loss = reduce(loss, "b t c -> b c", "sum")/camera_imask_sum

        # velocity loss
        camera_gt_v = camera_gt[:, 1:] - camera_gt[:, :-1]
        camera_res_v = camera_res[:, 1:] - camera_res[:, :-1]
        v_loss = self.loss_fn(camera_res_v, camera_gt_v, reduction="none")
        # remove the velocity between keyframe and history
        camera_imask_v = camera_imask.clone()
        camera_imask_v[:,self.history_len:self.history_len+1] *= 0
        v_loss = v_loss*camera_imask_v[:,1:]
        v_loss = reduce(v_loss, "b t c -> b c", "sum")/camera_imask_sum # should be camera_imask_sum - 1, here we use camera_imask_sum to avoid zero and nan


        # acceleration loss
        camera_gt_a = camera_gt_v[:, 1:] - camera_gt_v[:, :-1]
        camera_res_a = camera_res_v[:, 1:] - camera_res_v[:, :-1]
        a_loss = self.loss_fn(camera_res_a, camera_gt_a, reduction="none")
        # remove the acceleration between keyframe and history
        camera_imask_a = camera_imask.clone()
        camera_imask_a[:,self.history_len:self.history_len+2] *= 0
        a_loss = a_loss*camera_imask_a[:,2:]
        a_loss = reduce(a_loss, "b t c -> b c", "sum")/camera_imask_sum# should be camera_imask_sum - 2, here we use camera_imask_sum to avoid zero and nan

        # body attention loss
        unnormalized_motion = self.normalizer_pose.unnormalize(p_cond)#(N, seq, joint*3)
        reshape_motion = unnormalized_motion.reshape(unnormalized_motion.shape[0], unnormalized_motion.shape[1], -1, 3)#(N, seq, joint, 3)
        transpose_motion = reshape_motion.transpose(2,1).transpose(0,1)#(joint, N, seq, 3)
        if self.camera_format == 'polar':# polar coordinates
            out_camera_dis = self.normalizer_camera_dis.unnormalize(camera_res[:,:,:1])#(N, seq, 1)
            out_camera_pos = self.normalizer_camera_pos.unnormalize(camera_res[:,:,1:4])#(N, seq, 3)
            out_camera_rot = self.normalizer_camera_rot.unnormalize(camera_res[:,:,4:7])#(N, seq, 3)
            out_camera_fov = self.normalizer_camera_fov.unnormalize(camera_res[:,:,7:8])#(N, seq, 1)
        elif self.camera_format == 'centric':#camera centric representation/ Cartesian coordinates
            out_camera_rot = self.normalizer_camera_rot.unnormalize(camera_res[:,:,:3])#(N, seq, 3)
            out_camera_fov = self.normalizer_camera_fov.unnormalize(camera_res[:,:,3:4])#(N, seq, 1)
            out_camera_eye = self.normalizer_camera_eye.unnormalize(camera_res[:,:,4:7])#(N, seq, 3)

        b, s, _ = out_camera_rot.shape
        device = out_camera_rot.device
        t1 = torch.ones(b,s,1).to(device)
        t0 = torch.zeros(b,s,1).to(device)
        view = torch.ones(b,s,4,4) * torch.eye(4)
        view = view.to(device)
        if self.camera_format == 'polar':#polar coordinates
            camera_dis_expand = torch.cat([t0,t0, torch.abs(out_camera_dis)],dim=-1)
            view = torch_glm_translate(view,camera_dis_expand)
        rot = torch.ones(b,s,4,4) * torch.eye(4)
        rot = rot.to(device)
        rot = torch_glm_rotate(rot,out_camera_rot[:,:,1],torch.cat([t0,t1,t0], dim=-1))
        rot = torch_glm_rotate(rot,out_camera_rot[:,:,2],torch.cat([t0,t0,t1*(-1)], dim=-1))
        rot = torch_glm_rotate(rot,out_camera_rot[:,:,0],torch.cat([t1,t0,t0], dim=-1))
        view = torch.matmul(view,rot)

        if self.camera_format == 'polar':#polar coordinates
            out_camera_eye = view[:,:,3,:3] + out_camera_pos * torch.cat([t1,t1,t1*(-1)], dim=-1)
        out_camera_z = F.normalize(view[:,:,2,:3]*(-1),p=2,dim=-1)
        out_camera_y = F.normalize(view[:,:,1,:3],p=2,dim=-1)
        out_camera_x = F.normalize(view[:,:,0,:3],p=2,dim=-1)

        out_motion2eye = transpose_motion - out_camera_eye
        out_kps_yz = out_motion2eye - out_camera_x * torch.sum(out_motion2eye * out_camera_x, axis=-1, keepdims = True) 
        out_kps_xz = out_motion2eye - out_camera_y * torch.sum(out_motion2eye * out_camera_y, axis=-1, keepdims = True) 
        out_cos_y_z = torch.sum(out_kps_yz * out_camera_z, axis=-1)
        out_cos_x_z = torch.sum(out_kps_xz * out_camera_z, axis=-1)
        out_cos_fov = torch.cos(out_camera_fov*0.5/180 * math.pi)
        out_cos_fov = out_cos_fov.reshape(out_cos_fov.shape[0], out_cos_fov.shape[1])#(N, seq)
        
        out_diff_x = (out_cos_fov * torch.sqrt(torch.sum(out_kps_xz * out_kps_xz, axis=-1)) - out_cos_x_z).transpose(0,1).transpose(2,1)#(N, seq, joint)
        out_diff_y = (out_cos_fov * torch.sqrt(torch.sum(out_kps_yz * out_kps_yz, axis=-1)) - out_cos_y_z).transpose(0,1).transpose(2,1)

        in_ba_loss = F.relu(out_diff_x*b_mask)+F.relu(out_diff_y*b_mask)# inside camera view 
        in_ba_loss = in_ba_loss*camera_imask
        in_ba_loss = reduce(in_ba_loss, "b t c -> b c", "sum")/camera_imask_sum
        out_ba_loss = F.relu(out_diff_x*(b_mask-1)) * F.relu(out_diff_y*(b_mask-1))# outside camera view
        out_ba_loss = out_ba_loss*camera_imask
        out_ba_loss = reduce(out_ba_loss, "b t c -> b c", "sum")/camera_imask_sum
        losses = (
            self.w_loss  * loss.mean(),
            self.w_v_loss * v_loss.mean(),
            self.w_a_loss * a_loss.mean(),
            self.w_in_ba_loss * in_ba_loss.mean(),
            self.w_out_ba_loss * out_ba_loss.mean(),
        )
        return sum(losses), losses

    def load_datasets(self, opt):
        
        # load datasets
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
            train_dataset = DCMPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
                history_len = self.history_len,
                inference_len = self.inference_len,
                feature_type = opt.feature_type
            )
            test_dataset = DCMPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer_pose= train_dataset.normalizer_pose,
                normalizer_camera_dis= train_dataset.normalizer_camera_dis,
                normalizer_camera_pos= train_dataset.normalizer_camera_pos,
                normalizer_camera_rot= train_dataset.normalizer_camera_rot,
                normalizer_camera_fov= train_dataset.normalizer_camera_fov,
                normalizer_camera_eye= train_dataset.normalizer_camera_eye,
                force_reload=opt.force_reload,
                history_len = self.history_len,
                inference_len = self.inference_len,
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
        self.normalizer_camera_dis = test_dataset.normalizer_camera_dis
        self.normalizer_camera_pos = test_dataset.normalizer_camera_pos
        self.normalizer_camera_rot = test_dataset.normalizer_camera_rot
        self.normalizer_camera_fov = test_dataset.normalizer_camera_fov
        self.normalizer_camera_eye = test_dataset.normalizer_camera_eye

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
            batch_size=opt.batch_size,
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
            avg_vloss = 0
            avg_aloss = 0
            avg_in_ba_loss = 0
            avg_out_ba_loss = 0
            # train
            self.train()

            for step, (x_camera, x_camera_imask, x_bone_mask, pose_cond, music_cond, x_wavpath, x_pre_padding, x_suf_padding, x_start_frame, x_end_frame) in enumerate(
                load_loop(train_data_loader)
            ):
                if self.camera_format == 'polar':
                    # use camera polar representation dis-1 pos-3 rot-3 fov-1
                    x_camera = x_camera[:,:,:8]
                elif self.camera_format == 'centric':
                    # use camera centric representation rot-3 fov-1 eye-3
                    x_camera = x_camera[:,:,-7:]
                    
                x_camera_res = self.model(
                    x_camera, x_camera_imask, pose_cond, music_cond
                )

                total_loss, (loss, v_loss, a_loss, in_ba_loss, out_ba_loss) = self.calculate_losses(x_camera_res, x_camera, x_camera_imask, x_bone_mask, pose_cond)

                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_aloss += a_loss.detach().cpu().numpy()
                    avg_in_ba_loss += in_ba_loss.detach().cpu().numpy()
                    avg_out_ba_loss += out_ba_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.ema.update_model_average(
                            self.master_model, self.model
                        )
                    #log
                    log_dict = {
                        "Train Loss": avg_loss/ len(train_data_loader),
                        "V Loss": avg_vloss/ len(train_data_loader),
                        "A Loss": avg_aloss/ len(train_data_loader),
                        "InsideBodyAttention Loss": avg_in_ba_loss/ len(train_data_loader),
                        "OutsideBodyAttention Loss": avg_out_ba_loss/ len(train_data_loader),
                    }
                    wandb.log(log_dict)
            # Save model
            if (epoch % opt.save_interval) == 0:
                
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    ckpt = {
                        "ema_state_dict": self.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer_pose": self.normalizer_pose,
                        "normalizer_camera_dis": self.normalizer_camera_dis,
                        "normalizer_camera_pos": self.normalizer_camera_pos,
                        "normalizer_camera_rot": self.normalizer_camera_rot,
                        "normalizer_camera_fov": self.normalizer_camera_fov,
                        "normalizer_camera_eye": self.normalizer_camera_eye,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # generate a sample
                    render_count = 3
                    sample_batch_size = test_dataset.subsequence_end_index[render_count-1]
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x_camera, x_camera_imask, x_bone_mask, pose_cond, music_cond, x_wavpath, x_pre_padding, x_suf_padding, x_start_frame, x_end_frame) = next(iter(test_data_loader))
                    pose_cond = pose_cond.to(self.accelerator.device)
                    music_cond = music_cond.to(self.accelerator.device)

                    self.render_sample(
                        sample_batch_size,
                        pose_cond,
                        music_cond,
                        x_camera,
                        x_camera_imask,
                        x_wavpath,
                        x_pre_padding,
                        x_suf_padding,
                        x_start_frame,
                        x_end_frame,
                        self.normalizer_pose,
                        self.normalizer_camera_dis,
                        self.normalizer_camera_pos,
                        self.normalizer_camera_rot,
                        self.normalizer_camera_fov,
                        self.normalizer_camera_eye,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        render_videos=opt.render_videos,
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
            
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(self, sample_batch_size, pose_cond, music_cond, x_camera, x_camera_imask, x_wavpath,     
                        x_pre_padding, x_suf_padding, x_start_frame, x_end_frame, normalizer_pose,
                        normalizer_camera_dis, normalizer_camera_pos,
                        normalizer_camera_rot, normalizer_camera_fov,
                        normalizer_camera_eye, epoch, out_dir, render_videos=True, sound=True, is_train = True, use_gt_keyframe_parameters = False,inserted_keyframe = None):
        
        pose_cond = pose_cond.to(self.accelerator.device)
        music_cond = music_cond.to(self.accelerator.device)
        if self.camera_format == 'polar':
            x_camera = x_camera[:,:,:8]
        elif self.camera_format == 'centric':
            x_camera = x_camera[:,:,-7:]
        x_camera = x_camera.to(self.accelerator.device)
        x_camera_imask = x_camera_imask.to(self.accelerator.device)
        unmormalize_pose = normalizer_pose.unnormalize(pose_cond).detach().cpu()

        sample_camera_eye = None # result for rendering
        sample_camera_rot = None
        sample_camera_fov = None
        sample_pose_cond = None # pose condition for rendering
        normalized_sample_camera_condition = None
        i_sample = 0
        pbar = tqdm(total=sample_batch_size)
        while(i_sample < sample_batch_size):

            tmp_imask_len = int(torch.sum(x_camera_imask[i_sample]))

            if sample_camera_eye == None:# this sample is a new start
                if is_train:
                    sample_result = self.model.module(x_camera[i_sample:i_sample+1]*0 + x_camera[i_sample:i_sample+1,0:1], x_camera_imask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                    # x_camera[i_sample:i_sample+1]*0 + x_camera[i_sample:i_sample+1,0:1] is actually a padding zero procedure because for a new start, x_camera[i_sample:i_sample+1,0:1] is normalized zeros
                elif use_gt_keyframe_parameters:
                    if tmp_imask_len < self.inference_len:# include the end keyframe
                        tmp_camera_imask = x_camera_imask[i_sample:i_sample+1]
                        tmp_camera_imask[:,self.history_len+tmp_imask_len] += 1
                        sample_result = self.model.forward_with_keyframe(x_camera[i_sample:i_sample+1], tmp_camera_imask, pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                    else:# use the last frame within the inference_len as end keyframe
                        sample_result = self.model.forward_with_keyframe(x_camera[i_sample:i_sample+1], x_camera_imask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                else:
                    sample_result = self.model(x_camera[i_sample:i_sample+1]*0 + x_camera[i_sample:i_sample+1,0:1], x_camera_imask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
            else:
                if is_train: 
                    sample_result = self.model.module(normalized_sample_camera_condition[0:1], x_camera_imask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                elif use_gt_keyframe_parameters:
                    if tmp_imask_len < self.inference_len:# include the end keyframe
                        tmp_camera_imask = x_camera_imask[i_sample:i_sample+1]
                        tmp_camera_imask[:,self.history_len+tmp_imask_len] += 1
                        sample_result = self.model.forward_with_keyframe(torch.cat((normalized_sample_camera_condition[0:1,:self.history_len],x_camera[i_sample:i_sample+1,self.history_len:]),dim=1), tmp_camera_imask, pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                    else:# use the last frame within the inference_len as end keyframe
                        sample_result = self.model.forward_with_keyframe(torch.cat((normalized_sample_camera_condition[0:1,:self.history_len],x_camera[i_sample:i_sample+1,self.history_len:]),dim=1), x_camera_imask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
                else:
                    sample_result = self.model(normalized_sample_camera_condition[0:1], x_camera_imask[i_sample:i_sample+1], pose_cond[i_sample:i_sample+1], music_cond[i_sample:i_sample+1])
            
            normalized_sample_camera_condition =  torch.zeros_like(sample_result)

            # add history
            normalized_sample_camera_condition[:,:self.history_len,:] = sample_result[:,tmp_imask_len:self.history_len+tmp_imask_len,:].detach()
            
            sample_result = sample_result.detach().cpu()
            
            if self.camera_format == 'polar':
                tmp_sample_camera_dis = sample_result[:,:,:1]
                tmp_sample_camera_pos = sample_result[:,:,1:4]
                tmp_sample_camera_rot = sample_result[:,:,4:7]
                tmp_sample_camera_fov = sample_result[:,:,7:8]
                tmp_sample_camera_dis = normalizer_camera_dis.unnormalize(tmp_sample_camera_dis).detach().cpu()
                tmp_sample_camera_pos = normalizer_camera_pos.unnormalize(tmp_sample_camera_pos).detach().cpu()
                tmp_sample_camera_rot = normalizer_camera_rot.unnormalize(tmp_sample_camera_rot).detach().cpu()
                tmp_sample_camera_fov = normalizer_camera_fov.unnormalize(tmp_sample_camera_fov).detach().cpu()
                # calculate tmp_sample_camera_eye
                b, s, _ = tmp_sample_camera_rot.shape
                device = tmp_sample_camera_rot.device
                t1 = torch.ones(b,s,1).to(device)
                t0 = torch.zeros(b,s,1).to(device)
                view = torch.ones(b,s,4,4) * torch.eye(4)
                view = view.to(device)
                tmp_sample_camera_dis_expand = torch.cat([t0,t0,torch.abs(tmp_sample_camera_dis)],dim=-1)
                view = torch_glm_translate(view,tmp_sample_camera_dis_expand)
                rot = torch.ones(b,s,4,4) * torch.eye(4)
                rot = rot.to(device)
                rot = torch_glm_rotate(rot,tmp_sample_camera_rot[:,:,1],torch.cat([t0,t1,t0], dim=-1))
                rot = torch_glm_rotate(rot,tmp_sample_camera_rot[:,:,2],torch.cat([t0,t0,t1*(-1)], dim=-1))
                rot = torch_glm_rotate(rot,tmp_sample_camera_rot[:,:,0],torch.cat([t1,t0,t0], dim=-1))
                view = torch.matmul(view,rot)
                tmp_sample_camera_eye = view[:,:,3,:3] + tmp_sample_camera_pos * torch.cat([t1,t1,t1*(-1)], dim=-1)
            elif self.camera_format == 'centric':
                tmp_sample_camera_rot = sample_result[:,:,:3]
                tmp_sample_camera_fov = sample_result[:,:,3:4]
                tmp_sample_camera_eye = sample_result[:,:,4:7]
                tmp_sample_camera_rot = normalizer_camera_rot.unnormalize(tmp_sample_camera_rot).detach().cpu()
                tmp_sample_camera_fov = normalizer_camera_fov.unnormalize(tmp_sample_camera_fov).detach().cpu()
                tmp_sample_camera_eye = normalizer_camera_eye.unnormalize(tmp_sample_camera_eye).detach().cpu()


            # synthesize results
            if sample_camera_fov == None: # means start of a piece of test data
                sample_camera_rot = tmp_sample_camera_rot[0:1,self.history_len:self.history_len+tmp_imask_len,:]
                sample_camera_fov = tmp_sample_camera_fov[0:1,self.history_len:self.history_len+tmp_imask_len,:]
                sample_camera_eye = tmp_sample_camera_eye[0:1,self.history_len:self.history_len+tmp_imask_len,:]
                sample_pose_cond = unmormalize_pose[i_sample:i_sample+1,self.history_len:self.history_len+tmp_imask_len,:]
            else: # means not a start of a piece of test data
                sample_camera_rot = torch.cat([sample_camera_rot, tmp_sample_camera_rot[0:1,self.history_len:self.history_len+tmp_imask_len,:]], dim=1)
                sample_camera_fov = torch.cat([sample_camera_fov, tmp_sample_camera_fov[0:1,self.history_len:self.history_len+tmp_imask_len,:]], dim=1)
                sample_camera_eye = torch.cat([sample_camera_eye, tmp_sample_camera_eye[0:1,self.history_len:self.history_len+tmp_imask_len,:]], dim=1)
                sample_pose_cond = torch.cat([sample_pose_cond, unmormalize_pose[i_sample:i_sample+1,self.history_len:self.history_len+tmp_imask_len,:]], dim=1)
            
            # rendering
            if i_sample + 1 == sample_batch_size: # this is the last sub-sequence
                self.save_render(sample_camera_eye, sample_camera_rot, sample_camera_fov, sample_pose_cond, epoch, out_dir, x_wavpath[i_sample], render_videos, sound)
                # reset
                sample_camera_eye = None # result for rendering
                sample_camera_rot = None
                sample_camera_fov = None
                sample_pose_cond = None # pose condition for rendering
                normalized_sample_camera_condition = None
                i_sample += 1
            elif not x_wavpath[i_sample + 1] == x_wavpath[i_sample]:#here is an end for a piece of data
                self.save_render(sample_camera_eye, sample_camera_rot, sample_camera_fov, sample_pose_cond, epoch, out_dir, x_wavpath[i_sample], render_videos, sound)
                # reset
                sample_camera_eye = None # result for rendering
                sample_camera_rot = None
                sample_camera_fov = None
                sample_pose_cond = None # pose condition for rendering
                i_sample += 1
                normalized_sample_camera_condition = None
            else:#current sample not end
                i_sample += 1
            pbar.update(1)


    
    def save_render(self, c_eye, c_rot, c_fov, p_cond, epoch, render_out, wav_path, render_videos, sound):
        assert len(c_eye) == len(c_rot) == len(c_fov)
        assert len(c_eye) == len(p_cond)
        # calculate c_x,y,z
        b, s, _ = c_rot.shape
        device = c_rot.device
        t1 = torch.ones(b,s,1).to(device)
        t0 = torch.zeros(b,s,1).to(device)

        view = torch.ones(b,s,4,4) * torch.eye(4)
        view = view.to(device)
        # use centric representation
        rot = torch.ones(b,s,4,4) * torch.eye(4)
        rot = rot.to(device)
        rot = torch_glm_rotate(rot,c_rot[:,:,1],torch.cat([t0,t1,t0], dim=-1))
        rot = torch_glm_rotate(rot,c_rot[:,:,2],torch.cat([t0,t0,t1*(-1)], dim=-1))
        rot = torch_glm_rotate(rot,c_rot[:,:,0],torch.cat([t1,t0,t0], dim=-1))

        view = torch.matmul(view,rot)
        c_z = F.normalize(view[:,:,2,:3]*(-1),p=2,dim=-1)
        c_y = F.normalize(view[:,:,1,:3],p=2,dim=-1)
        c_x = F.normalize(view[:,:,0,:3],p=2,dim=-1)
        # detect bonemask
        reshape_motion = p_cond.reshape(p_cond.shape[0], p_cond.shape[1], -1, 3)#(batch, seq, joint, 3)
        transpose_motion = reshape_motion.transpose(2,1).transpose(0,1)#(joint, batch, seq, 3)

        motion2eye = transpose_motion - c_eye#(joint, batch, seq, 3)
        kps_yz = motion2eye - c_x * torch.sum(motion2eye * c_x, axis=-1, keepdims = True) #(joint, batch, seq, 3)
        kps_xz = motion2eye - c_y * torch.sum(motion2eye * c_y, axis=-1, keepdims = True) 
        cos_y_z = torch.sum(kps_yz * c_z, axis=-1)#(batch, joint, seq)
        cos_x_z = torch.sum(kps_xz * c_z, axis=-1)
        cos_fov = torch.cos(c_fov*0.5/180 * math.pi)
        cos_fov = cos_fov.reshape(cos_fov.shape[0], cos_fov.shape[1])#(batch, seq)

        diff_x = (cos_x_z - cos_fov * torch.sqrt(torch.sum(kps_xz * kps_xz, axis=-1))).transpose(0,1).transpose(2,1)#(batch, seq, joint)
        diff_y = (cos_y_z - cos_fov * torch.sqrt(torch.sum(kps_yz * kps_yz, axis=-1))).transpose(0,1).transpose(2,1)

        diff_x[diff_x >= 0] = 1
        diff_x[diff_x < 0] = 0 
        diff_y[diff_y >= 0] = 1
        diff_y[diff_y < 0] = 0
        
        bone_mask = diff_x + diff_y
        bone_mask[bone_mask < 2] = 0
        bone_mask[bone_mask >= 2] = 1
        # camera_name
        norm_wav_path = os.path.normpath(wav_path)# wav_path is like "DCM++/Test/Audio/a3_1_C.wav"
        pathparts = norm_wav_path.split(os.sep)
        camera_name = 'c' + pathparts[-1][1:-6] + '.json'
        # save and render
        MotionCameraRenderSave(
            p_cond[0],
            c_eye[0],
            c_z[0],
            c_y[0],
            c_x[0],
            c_rot[0],
            c_fov[0],
            bone_mask[0],
            epoch=f"e{epoch}",
            out_dir=render_out,
            audio_path=wav_path,
            camera_name=camera_name,
            render_videos=render_videos,
            sound=sound
        )
                
