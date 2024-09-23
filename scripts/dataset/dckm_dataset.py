# load dance, camera keyframe, and music dataset
import os
import glob
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
from .preprocess import Normalizer, vectorize_many

def GetIndex(e):
    e_basename = os.path.splitext(os.path.basename(e))[0]
    e_name = e_basename.split('.')[0]
    e_name_split = e_name[1:].split('_')
    m_index = e_name_split[0]
    if len(e_name_split) >= 2:
        if e_name_split[1] == 'C' or e_name_split[1] == 'E' or e_name_split[1] == 'J' or e_name_split[1] == 'K' or e_name_split[1] == 'kps3D':
            s_index = '0'
        else:
            s_index = e_name_split[1]
    else:
        s_index = '0'
    return m_index+'_'+s_index

def GetIndexBM(e):#BM and KM
    e_basename = os.path.splitext(os.path.basename(e))[0]
    e_name = e_basename.split('.')[0]
    e_name_split = e_name[2:].split('_')
    m_index = e_name_split[0]
    if len(e_name_split) >= 2:
        s_index = e_name_split[1]
    else:
        s_index = '0'
    return m_index+'_'+s_index

class DCKMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "aist",
        normalizer_pose: Any = None,
        history_len: int = -1,
        inference_len: int = -1,
        stride_len: int = -1,
        force_reload: bool = False,
        loaded_normalizer: bool = False,
    ):
        self.data_path = data_path
        self.data_fps = 30

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer_pose = normalizer_pose

        self.history_len = history_len
        self.inference_len = inference_len
        self.stride_len = stride_len


        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        # save normalizer
        if not train or loaded_normalizer:
            pickle.dump(
                normalizer_pose, open(os.path.join(backup_path, "normalizer_pose.pkl"), "wb")
            )
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_dckmpp()  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pose: {data['pos'].shape}, camera_keyframe: {data['camera_keyframe'].shape}, padding_mask: {data['padding_mask'].shape},  acoustic_feature: {data['feature'].shape}"
        )

        # process data
        motion_seq, camera_keyframe_seq, padding_mask = self.process_dataset(data["pos"], data['camera_keyframe'], data['padding_mask'])
        
        
        self.data = {
            "motion": motion_seq,
            "camera_keyframe": camera_keyframe_seq,
            "padding_mask": padding_mask,
            "acoustic_feature": data['feature'],
            "wavpath": data['wavpath']
        }
        assert len(motion_seq) == len(data['padding_mask']) == len(data['wavpath']) == len(data['camera_keyframe'])
        self.length = len(motion_seq)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data["camera_keyframe"][idx].long(), self.data["padding_mask"][idx], self.data["motion"][idx], self.data["acoustic_feature"][idx], self.data["wavpath"][idx]

    def load_dckmpp(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "Train" if self.train else "Test"
        )

        # Structure:
        # DCM++
        #   |- Train
        #   |    |- aist_feats_long
        #   |    |- Audio
        #   |    |- CameraCentric
        #   |    |- CameraKeyframe
        #   |    |- Keypoints3Ds
        #   |    |- Simplified_MotionGlobalTransform
        #   |- Test
        #   ...

        motion_path = os.path.join(split_data_path, "Keypoints3D")
        camera_path = os.path.join(split_data_path, "CameraCentric")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats_long")
        wav_path = os.path.join(split_data_path, f"Audio")
        
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.json")),key=GetIndex)
        cameras = sorted(glob.glob(os.path.join(camera_path, "*.json")),key=GetIndex)
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")),key=GetIndex)
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")),key=GetIndex)
        assert len(motions) == len(cameras) == len(features) == len(wavs)
        number_data = len(motions)
        # print(features)
        

        # stack the motions, cameras and features together
        all_pos = []
        all_c_kmask = []
        all_padding_mask = []# padding mask
        all_feature = []
        all_names = []
        all_wavs = []
        # print(sound_path)
        self.subsequence_end_index = [] # denote that for each raw data, the corresponding subsequences end index. works for test dataset

        tmp_end_index = 0
        for i_data in tqdm(range(number_data)):
            motion = motions[i_data]
            camera = cameras[i_data]
            feature = features[i_data]
            wav = wavs[i_data]
            
            m_name = os.path.splitext(os.path.basename(motion))[0]#m*_kps3D
            c_name = os.path.splitext(os.path.basename(camera))[0]#c*
            f_name = os.path.splitext(os.path.basename(feature))[0]#a*_(C,E,K,J)
            w_name = os.path.splitext(os.path.basename(wav))[0]#a*_(C,E,K,J)
            # make sure name is matching
            assert m_name[1:-6] == c_name[1:] == f_name[1:-2] == w_name[1:-2]

            
            # load camera
            with open(camera, 'r') as cf:
                camera_data = json.load(cf)
            
            c_kmask = np.array(camera_data['KeyframeMask'])
            c_kmask[-1] = 1.0# set last frame to be 1
            c_kpos = np.array(camera_data['KeyframePos'])
            cur_framenum = len(c_kmask)
            # load motion
            with open(motion, 'r') as mf:
                motion_data = json.load(mf)
            pos = np.array(motion_data["Keypoints3D"])[:cur_framenum,:]#(frame_num, 60*3)

            # load the feature
            feature_data = np.load(feature)[:cur_framenum,:]
            
            # cut into training sub-sequence, do padding here
            cur_frame_index = 0
            cur_frame_sum = len(c_kmask)
            while cur_frame_index < cur_frame_sum:
                i_pos = np.zeros((self.history_len+self.inference_len, pos.shape[1]),dtype=np.float32)
                i_c_kmask = np.zeros(self.history_len+self.inference_len,dtype=np.float32)
                i_padding_mask = np.zeros(self.history_len+self.inference_len,dtype=np.float32)
                i_feature = np.zeros((self.history_len+self.inference_len, feature_data.shape[1]),dtype=np.float32)
                # history period
                if cur_frame_index < self.history_len: # need pre padding
                    i_pos[self.history_len - cur_frame_index:self.history_len] = pos[:cur_frame_index]
                    i_c_kmask[self.history_len - cur_frame_index:self.history_len] = c_kmask[:cur_frame_index]
                    i_feature[self.history_len - cur_frame_index:self.history_len] = feature_data[:cur_frame_index]
                    i_padding_mask[self.history_len - cur_frame_index:self.history_len] += 1
                else:# history do not need padding
                    i_pos[:self.history_len] = pos[cur_frame_index-self.history_len:cur_frame_index]
                    i_c_kmask[:self.history_len] = c_kmask[cur_frame_index-self.history_len:cur_frame_index]
                    i_feature[:self.history_len] = feature_data[cur_frame_index-self.history_len:cur_frame_index]
                    i_padding_mask[:self.history_len] += 1
                # inference period
                if cur_frame_sum - cur_frame_index < self.inference_len:
                    i_pos[self.history_len:self.history_len+cur_frame_sum-cur_frame_index] = pos[cur_frame_index:]
                    i_c_kmask[self.history_len:self.history_len+cur_frame_sum-cur_frame_index] = c_kmask[cur_frame_index:]
                    i_feature[self.history_len:self.history_len+cur_frame_sum-cur_frame_index] = feature_data[cur_frame_index:]
                    i_padding_mask[self.history_len:self.history_len+cur_frame_sum-cur_frame_index] += 1
                else:
                    i_pos[self.history_len:] = pos[cur_frame_index:cur_frame_index+self.inference_len]
                    i_c_kmask[self.history_len:] = c_kmask[cur_frame_index:cur_frame_index+self.inference_len]
                    i_feature[self.history_len:] = feature_data[cur_frame_index:cur_frame_index+self.inference_len]
                    i_padding_mask[self.history_len:] += 1
                all_pos.append(i_pos)
                all_c_kmask.append(i_c_kmask)
                all_padding_mask.append(i_padding_mask)
                all_feature.append(i_feature)
                all_wavs.append(wav)
                cur_frame_index += self.stride_len
                tmp_end_index += 1
            self.subsequence_end_index.append(tmp_end_index)
        all_pos = np.array(all_pos)  # N x seq x (joint * 3), seq = history+inference
        all_c_kmask = np.array(all_c_kmask)  # N x seq x 1
        all_padding_mask = np.array(all_padding_mask)  # N x seq x 1
        all_feature = np.array(all_feature).astype(dtype=np.float32) # N x seq x feature dim
        # downsample the motions to the data fps
        data = {"pos": all_pos, "camera_keyframe": all_c_kmask, "padding_mask": all_padding_mask, "feature": all_feature, "wavpath": all_wavs}

        return data

    def process_dataset(self, kps_pos, camera_kf, pd_mask):
        kps_pos = torch.Tensor(np.array(kps_pos,dtype='float32'))
        camera_kf = torch.Tensor(np.array(camera_kf,dtype='float32'))
        pd_mask = torch.Tensor(np.array(pd_mask,dtype='float32'))
        # now, flatten everything into: batch x sequence x [...]
        pose_vec_input = vectorize_many([kps_pos]).float().detach()
        camera_kf_vec_input = vectorize_many([camera_kf]).float().detach()
        pd_mask_vec_input = vectorize_many([pd_mask]).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer_pose = Normalizer(pose_vec_input)
        else:
            assert self.normalizer_pose is not None
        pose_vec_input = self.normalizer_pose.normalize(pose_vec_input)

        assert not torch.isnan(pose_vec_input).any()
        data_name = "Train" if self.train else "Test"


        print(f"{data_name} Dataset Motion Features Dim: {pose_vec_input.shape}")

        return pose_vec_input, camera_kf_vec_input, pd_mask_vec_input

