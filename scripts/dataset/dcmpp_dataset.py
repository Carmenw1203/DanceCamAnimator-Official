#load dance camera music dataset
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

class DCMPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "aist",
        normalizer_pose: Any = None,
        normalizer_camera_dis: Any = None,
        normalizer_camera_pos: Any = None,
        normalizer_camera_rot: Any = None,
        normalizer_camera_fov: Any = None,
        normalizer_camera_eye: Any = None,
        history_len: int = -1,
        inference_len: int = -1,
        force_reload: bool = False,
        loaded_normalizer: bool = False,
        generated_keyframemask_dir: str = ""
    ):
        self.data_path = data_path
        self.generated_keyframemask_dir = generated_keyframemask_dir
        self.data_fps = 30

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer_pose = normalizer_pose
        self.normalizer_camera_dis = normalizer_camera_dis
        self.normalizer_camera_pos = normalizer_camera_pos
        self.normalizer_camera_rot = normalizer_camera_rot
        self.normalizer_camera_fov = normalizer_camera_fov
        self.normalizer_camera_eye = normalizer_camera_eye

        self.history_len = history_len
        self.inference_len = inference_len

        self.subsequence_end_index = [] # denote that for each raw data, the corresponding subsequences end index. works for test dataset

        # denote whether this data piece starts inserted keyframe, used for render_sample function
        self.inserted_keyframe = []

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        # save normalizer
        if not train or loaded_normalizer:
            pickle.dump(
                normalizer_pose, open(os.path.join(backup_path, "normalizer_pose.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_dis, open(os.path.join(backup_path, "normalizer_camera_dis.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_pos, open(os.path.join(backup_path, "normalizer_camera_pos.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_rot, open(os.path.join(backup_path, "normalizer_camera_rot.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_fov, open(os.path.join(backup_path, "normalizer_camera_fov.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_eye, open(os.path.join(backup_path, "normalizer_camera_eye.pkl"), "wb")
            )
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print(force_reload,pickle_name,backup_path,os.listdir(backup_path))
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
                self.subsequence_end_index = data["subsequence_end_index"]
        elif not self.generated_keyframemask_dir == "":# using generated keyframe mask, not conflict with load cached dataset when using different backup_paths
        # if not self.generated_keyframemask_dir == "":# using generated keyframe mask
            print("Loading dataset with generated keyframe mask")
            data = self.load_dcmpp(use_gt_kmask = False)  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        else:
            print("Loading dataset...")
            data = self.load_dcmpp()  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pose: {data['pos'].shape}, camera_dis: {data['c_dis'].shape}, camera_pos: {data['c_pos'].shape}, camera_rot: {data['c_rot'].shape}, camera_fov: {data['c_fov'].shape}, camera_eye: {data['c_eye'].shape}, camera_inference_mask: {data['c_imask'].shape}, bone_mask: {data['b_mask'].shape}, acoustic_feature: {data['feature'].shape}, subsequence_end_index: {self.subsequence_end_index}"
        )

        # process data
        motion_n_camera, bone_mask = self.process_dataset(data["pos"], data['c_dis'], data['c_pos'], data['c_rot'], data['c_fov'], data['c_eye'], data['c_imask'], data['b_mask'])
        
        
        self.data = {
            "motion": motion_n_camera[:,:,:60*3],
            "camera": motion_n_camera[:,:,60*3:-1],
            "camera_imask": motion_n_camera[:,:,-1:],
            "bone_mask": bone_mask,
            "acoustic_feature": data['feature'],
            "wavpath": data['wavpath'],
            "pre_padding": data['pre_padding'],
            "suf_padding": data['suf_padding'],
            "start_frame": data['start_frame'],
            "end_frame": data['end_frame'],
        }
        assert len(motion_n_camera) == len(data['b_mask']) == len(data['wavpath'])
        # print(data['feature'].dtype)
        self.length = len(motion_n_camera)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data["camera"][idx], self.data["camera_imask"][idx], self.data["bone_mask"][idx], self.data["motion"][idx], self.data["acoustic_feature"][idx], self.data["wavpath"][idx], self.data["pre_padding"][idx], self.data["suf_padding"][idx], self.data["start_frame"][idx], self.data["end_frame"][idx]


    def load_dcmpp(self, use_gt_kmask = True):
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
        bone_mask_path = os.path.join(split_data_path, "BoneMask")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats_long")
        wav_path = os.path.join(split_data_path, f"Audio")
        
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.json")),key=GetIndex)
        cameras = sorted(glob.glob(os.path.join(camera_path, "*.json")),key=GetIndex)
        bone_masks = sorted(glob.glob(os.path.join(bone_mask_path, "*.json")),key=GetIndexBM)
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")),key=GetIndex)
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")),key=GetIndex)
        if not use_gt_kmask: # not use groundtruth data and load generated keyframe mask
            keyframe_masks = sorted(glob.glob(os.path.join(self.generated_keyframemask_dir, "*.json")),key=GetIndexBM)
            assert len(motions) == len(cameras) == len(bone_masks) == len(features) == len(wavs) == len(keyframe_masks)
        else:
            assert len(motions) == len(cameras) == len(bone_masks) == len(features) == len(wavs)
        number_data = len(motions)
        

        # stack the motions, cameras and features together
        all_pos = []# human pose
        all_c_dis = []# camera motion parameters
        all_c_pos = []
        all_c_rot = []
        all_c_fov = []
        all_c_eye = []
        all_b_mask = []# bone mask
        all_c_imask = []# camera inference mask, 1 for current keyframe and frames before next keyframe, 0 for others
        all_c_kpos = []# postion of camera keyframe
        all_feature = []# music feature
        all_wavs = []# music paths
        all_pre_padding = []# prefix padding zeros number
        all_suf_padding = []# suffix padding zeros number
        all_start_frame = []# start frame in original data
        all_end_frame = []# end frame in original data
        
        tmp_end_index = 0
        print('number_data: ',number_data)
        for i_data in tqdm(range(number_data)):
            motion = motions[i_data]
            camera = cameras[i_data]
            bone_mask = bone_masks[i_data]
            feature = features[i_data]
            wav = wavs[i_data]
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            c_name = os.path.splitext(os.path.basename(camera))[0]
            bm_name = os.path.splitext(os.path.basename(bone_mask))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            if not use_gt_kmask:
                keyframe_mask = keyframe_masks[i_data]
                km_name = os.path.splitext(os.path.basename(keyframe_mask))[0]
                assert m_name[1:-6] == c_name[1:] == bm_name[2:] == f_name[1:-2] == w_name[1:-2] == km_name[2:]
            else:
                assert m_name[1:-6] == c_name[1:] == bm_name[2:] == f_name[1:-2] == w_name[1:-2]

            
            # load camera
            with open(camera, 'r') as cf:
                camera_data = json.load(cf)
            c_dis = np.array(camera_data['Distance'])
            c_pos = np.array(camera_data['Position'])
            c_rot = np.array(camera_data['Rotation'])
            c_fov = np.array(camera_data['Fov'])
            c_eye = np.array(camera_data['camera_eye'])
            if not use_gt_kmask:
                with open(keyframe_mask, 'r') as kmf:
                    keyframe_mask_data = json.load(kmf)
                c_kmask = np.array(keyframe_mask_data['KeyframeMask'])
                c_kpos = np.array(keyframe_mask_data['KeyframePos'])#keyframe position
            else:
                c_kmask = np.array(camera_data['KeyframeMask'])
                c_kpos = np.array(camera_data['KeyframePos'])

            assert len(c_dis) == len(c_kmask)
            # enforce the start and end frames to be keyframes
            if not c_kmask[0] == 1:
                c_kmask[0] = 1
                c_kpos = np.insert(c_kpos,0,0)
            if not c_kmask[-1] == 1:
                c_kmask[-1] = 1
                c_kpos = np.insert(c_kpos,len(c_kpos),len(c_kmask)-1)

            # add new keyframe to preprocess data which has longer keyframe interval(>inference length)
            new_c_kpos = []
            cur_c_kpos = 0
            new_c_kpos.append(cur_c_kpos)
            self.inserted_keyframe.append(0)# for normal keyframe
            for ck_pos_index in range(1,len(c_kpos)):
                while(c_kpos[ck_pos_index] - cur_c_kpos) > self.inference_len:
                    cur_c_kpos += self.inference_len
                    new_c_kpos.append(cur_c_kpos)
                    self.inserted_keyframe.append(1)# for inserted keyframe
                cur_c_kpos = c_kpos[ck_pos_index]
                new_c_kpos.append(cur_c_kpos)
                self.inserted_keyframe.append(0)# for normal keyframe
            c_kpos = np.array(new_c_kpos.copy())
            
            # check if all camera keyframe positions are in order and no interval between adjacent keyframes are bigger than self.inference_len
            for ck_pos_index in range(1,len(c_kpos)):
                if c_kpos[ck_pos_index] - c_kpos[ck_pos_index - 1] < 0 or c_kpos[ck_pos_index] - c_kpos[ck_pos_index - 1] > self.inference_len:
                    sys.exit(0)
            cur_framenum = len(c_eye)
            # load motion
            with open(motion, 'r') as mf:
                motion_data = json.load(mf)
            pos = np.array(motion_data["Keypoints3D"])[:cur_framenum,:]#(frame_num, 60*3)
            # load bone mask
            with open(bone_mask, 'r') as bmf:
                bone_mask_data = json.load(bmf)
            b_mask = np.array(bone_mask_data['bone_mask'])[:cur_framenum,:]
            # load the feature
            feature_data = np.load(feature)[:cur_framenum,:]
            
            # cut into training sub-sequence,[padding zeros,history camera,keyframe0:keyframe1, padding zeros] do padding here
            
            for ck_pos_index in range(len(c_kpos)):
                i_ck_pos = c_kpos[ck_pos_index]
                if ck_pos_index + 1 < len(c_kpos):
                    next_ck_pos = c_kpos[ck_pos_index+1]
                else:
                    next_ck_pos = -1

                i_pos = np.zeros((self.history_len+self.inference_len, pos.shape[1]),dtype=np.float32)
                i_c_dis = np.zeros(self.history_len+self.inference_len,dtype=np.float32)
                i_c_pos = np.zeros((self.history_len+self.inference_len, 3),dtype=np.float32)
                i_c_rot = np.zeros((self.history_len+self.inference_len, 3),dtype=np.float32)
                i_c_fov = np.zeros(self.history_len+self.inference_len,dtype=np.float32)
                i_c_eye = np.zeros((self.history_len+self.inference_len, 3),dtype=np.float32)
                i_c_imask = np.zeros(self.history_len+self.inference_len,dtype=np.float32)
                i_b_mask = np.zeros((self.history_len+self.inference_len, b_mask.shape[1]),dtype=np.float32)
                i_feature = np.zeros((self.history_len+self.inference_len, feature_data.shape[1]),dtype=np.float32)
                # i_wavpath = wav
                i_pre_padding = 0 # prefix padding length
                i_suf_padding = 0 # suffix padding length
                i_start_frame = -1 # start frame in original data
                i_end_frame = -1 # end frame in original data
                # history period
                if i_ck_pos < self.history_len:#history need padding
                    i_pos[self.history_len - i_ck_pos:self.history_len] = pos[:i_ck_pos]
                    i_c_dis[self.history_len - i_ck_pos:self.history_len] = c_dis[:i_ck_pos]
                    i_c_pos[self.history_len - i_ck_pos:self.history_len] = c_pos[:i_ck_pos]
                    i_c_rot[self.history_len - i_ck_pos:self.history_len] = c_rot[:i_ck_pos]
                    i_c_fov[self.history_len - i_ck_pos:self.history_len] = c_fov[:i_ck_pos]
                    i_c_eye[self.history_len - i_ck_pos:self.history_len] = c_eye[:i_ck_pos]
                    # here i_c_imask keep zeros
                    i_b_mask[self.history_len - i_ck_pos:self.history_len] = b_mask[:i_ck_pos]
                    i_feature[self.history_len - i_ck_pos:self.history_len] = feature_data[:i_ck_pos]
                    i_pre_padding = self.history_len - i_ck_pos
                    i_start_frame = 0
                else:# history do not need padding
                    i_pos[:self.history_len] = pos[i_ck_pos-self.history_len:i_ck_pos]
                    i_c_dis[:self.history_len] = c_dis[i_ck_pos-self.history_len:i_ck_pos]
                    i_c_pos[:self.history_len] = c_pos[i_ck_pos-self.history_len:i_ck_pos]
                    i_c_rot[:self.history_len] = c_rot[i_ck_pos-self.history_len:i_ck_pos]
                    i_c_fov[:self.history_len] = c_fov[i_ck_pos-self.history_len:i_ck_pos]
                    i_c_eye[:self.history_len] = c_eye[i_ck_pos-self.history_len:i_ck_pos]
                    # here i_c_imask keep zeros
                    i_b_mask[:self.history_len] = b_mask[i_ck_pos-self.history_len:i_ck_pos]
                    i_feature[:self.history_len] = feature_data[i_ck_pos-self.history_len:i_ck_pos]
                    i_pre_padding = 0
                    i_start_frame = i_ck_pos - self.history_len
                
                # inference period
                if len(c_eye)-i_ck_pos < self.inference_len:
                    i_pos[self.history_len:self.history_len+len(c_eye)-i_ck_pos] = pos[i_ck_pos:]
                    i_c_dis[self.history_len:self.history_len+len(c_eye)-i_ck_pos] = c_dis[i_ck_pos:]
                    i_c_pos[self.history_len:self.history_len+len(c_eye)-i_ck_pos] = c_pos[i_ck_pos:]
                    i_c_rot[self.history_len:self.history_len+len(c_eye)-i_ck_pos] = c_rot[i_ck_pos:]
                    i_c_fov[self.history_len:self.history_len+len(c_eye)-i_ck_pos] = c_fov[i_ck_pos:]
                    i_c_eye[self.history_len:self.history_len+len(c_eye)-i_ck_pos] = c_eye[i_ck_pos:]
                    # we will process i_c_imask latter
                    i_b_mask[self.history_len:self.history_len+len(c_eye)-i_ck_pos] = b_mask[i_ck_pos:]
                    i_feature[self.history_len:self.history_len+len(c_eye)-i_ck_pos] = feature_data[i_ck_pos:]
                    i_suf_padding = self.inference_len - len(c_eye) + i_ck_pos
                    i_end_frame = len(c_eye) - 1
                else:
                    i_pos[self.history_len:] = pos[i_ck_pos:i_ck_pos+self.inference_len]
                    i_c_dis[self.history_len:] = c_dis[i_ck_pos:i_ck_pos+self.inference_len]
                    i_c_pos[self.history_len:] = c_pos[i_ck_pos:i_ck_pos+self.inference_len]
                    i_c_rot[self.history_len:] = c_rot[i_ck_pos:i_ck_pos+self.inference_len]
                    i_c_fov[self.history_len:] = c_fov[i_ck_pos:i_ck_pos+self.inference_len]
                    i_c_eye[self.history_len:] = c_eye[i_ck_pos:i_ck_pos+self.inference_len]
                    # we will process i_c_imask latter
                    i_b_mask[self.history_len:] = b_mask[i_ck_pos:i_ck_pos+self.inference_len]
                    i_feature[self.history_len:] = feature_data[i_ck_pos:i_ck_pos+self.inference_len]
                    i_suf_padding = 0
                    i_end_frame = i_ck_pos + self.inference_len - 1
                # inference mask
                if next_ck_pos == -1:#have no next keyframe
                    i_c_imask[self.history_len:self.history_len+len(c_eye)-i_ck_pos] += 1
                else:
                    i_c_imask[self.history_len:self.history_len+next_ck_pos-i_ck_pos] += 1
                all_pos.append(i_pos)
                all_c_dis.append(i_c_dis)
                all_c_pos.append(i_c_pos)
                all_c_rot.append(i_c_rot)
                all_c_fov.append(i_c_fov)
                all_c_eye.append(i_c_eye)
                all_c_imask.append(i_c_imask)
                all_b_mask.append(i_b_mask)
                all_feature.append(i_feature)
                all_wavs.append(wav)
                all_pre_padding.append(i_pre_padding)
                all_suf_padding.append(i_suf_padding)
                all_start_frame.append(i_start_frame)
                all_end_frame.append(i_end_frame)
                tmp_end_index += 1
            self.subsequence_end_index.append(tmp_end_index)
            
        all_pos = np.array(all_pos)  # N x seq x (joint * 3), seq = history+inference
        all_c_dis = np.array(all_c_dis)  # N x seq x 1
        all_c_pos = np.array(all_c_pos)  # N x seq x 3
        all_c_rot = np.array(all_c_rot)  # N x seq x 3
        all_c_fov = np.array(all_c_fov)  # N x seq x 1
        all_c_eye = np.array(all_c_eye)  # N x seq x 3
        all_c_imask = np.array(all_c_imask) # N x seq x 1
        all_b_mask = np.array(all_b_mask)  # N x seq x joint
        all_feature = np.array(all_feature).astype(dtype=np.float32) # N x seq x feature dim
        # downsample the motions to the data fps
        data = {"pos": all_pos, "c_dis": all_c_dis, "c_pos": all_c_pos, "c_rot": all_c_rot, "c_fov": all_c_fov, "c_eye": all_c_eye, "c_imask": all_c_imask, "b_mask": all_b_mask, "feature": all_feature, "wavpath": all_wavs, "pre_padding": all_pre_padding, "suf_padding": all_suf_padding, "start_frame": all_start_frame, "end_frame": all_end_frame, "subsequence_end_index": self.subsequence_end_index}
        # print(self.subsequence_end_index)

        return data

    def process_dataset(self, kps_pos, camera_dis, camera_pos, camera_rot, camera_fov, camera_eye, camera_imask, b_mask):
        
        kps_pos = torch.Tensor(np.array(kps_pos,dtype='float32'))
        camera_dis = torch.Tensor(np.array(camera_dis,dtype='float32'))
        camera_pos = torch.Tensor(np.array(camera_pos,dtype='float32'))
        camera_rot = torch.Tensor(np.array(camera_rot,dtype='float32'))
        camera_fov = torch.Tensor(np.array(camera_fov,dtype='float32'))
        camera_eye = torch.Tensor(np.array(camera_eye,dtype='float32'))
        camera_imask = torch.Tensor(np.array(camera_imask,dtype='float32'))
        b_mask = torch.Tensor(np.array(b_mask,dtype='float32'))
        # now, flatten everything into: batch x sequence x [...]
        pose_vec_input = vectorize_many([kps_pos]).float().detach()
        camera_dis_vec_input = vectorize_many([camera_dis]).float().detach()
        camera_pos_vec_input = vectorize_many([camera_pos]).float().detach()
        camera_rot_vec_input = vectorize_many([camera_rot]).float().detach()
        camera_fov_vec_input = vectorize_many([camera_fov]).float().detach()
        camera_eye_vec_input = vectorize_many([camera_eye]).float().detach()
        camera_imask_vec_input = vectorize_many([camera_imask]).float().detach()
        b_mask_vec_input = vectorize_many([b_mask]).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer_pose = Normalizer(pose_vec_input)
            self.normalizer_camera_dis = Normalizer(camera_dis_vec_input)
            self.normalizer_camera_pos = Normalizer(camera_pos_vec_input)
            self.normalizer_camera_rot = Normalizer(camera_rot_vec_input)
            self.normalizer_camera_fov = Normalizer(camera_fov_vec_input)
            self.normalizer_camera_eye = Normalizer(camera_eye_vec_input)
        else:
            assert self.normalizer_pose is not None
            assert self.normalizer_camera_dis is not None
            assert self.normalizer_camera_pos is not None
            assert self.normalizer_camera_rot is not None
            assert self.normalizer_camera_fov is not None
            assert self.normalizer_camera_eye is not None
        pose_vec_input = self.normalizer_pose.normalize(pose_vec_input)
        camera_dis_vec_input = self.normalizer_camera_dis.normalize(camera_dis_vec_input)
        camera_pos_vec_input = self.normalizer_camera_pos.normalize(camera_pos_vec_input)
        camera_rot_vec_input = self.normalizer_camera_rot.normalize(camera_rot_vec_input)
        camera_fov_vec_input = self.normalizer_camera_fov.normalize(camera_fov_vec_input)
        camera_eye_vec_input = self.normalizer_camera_eye.normalize(camera_eye_vec_input)

        pose_camera_vec_input = torch.cat([pose_vec_input, camera_dis_vec_input, camera_pos_vec_input, camera_rot_vec_input, camera_fov_vec_input, camera_eye_vec_input,camera_imask_vec_input], dim = 2)
        assert not torch.isnan(pose_camera_vec_input).any()
        data_name = "Train" if self.train else "Test"


        print(f"{data_name} Dataset Motion Features Dim: {pose_camera_vec_input.shape}")

        return pose_camera_vec_input,b_mask_vec_input

