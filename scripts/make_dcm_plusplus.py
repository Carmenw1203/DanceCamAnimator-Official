# In order to better maintain the feature of keyframes, we make DCM++ from DCM by splitting the whole dataset without cutting the training data
# This is different from splitting method of original method of DCM
# But we still use the split files of DCM to keep the same distribution of training, test and validation sets


import os
import json
import shutil
import random
import argparse
from tqdm import tqdm
import librosa as lr
import numpy as np
import soundfile as sf
from pathlib import Path
import math
# from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from audio_extraction.baseline_features import extract_folder as aist_extract
from my_utils.slice import GlobalTransform2Keypoints, GlobalTransform2Keypoints_File
from my_utils.detect_bone_mask import DetactBoneMask
fps = 30

def FrameTimeFunc(e):
    return e["FrameTime"]



def GetSplitIndex(e):
    
    e_name_split = e[1:].split('_')
    r_index = e_name_split[1] #raw data id
    if len(e_name_split) >= 3:
        s_index = e_name_split[-1]
    else:
        s_index = '0'
    return r_index+'_'+s_index

def MakeDCMPlusPlus(args):

    with open(args.split_record,'r') as sf:
        split_dict = json.load(sf)

    with open(args.split_train, 'r') as trainf:
        train_list = json.load(trainf)
    with open(args.split_validation, 'r') as valf:
        validation_list = json.load(valf)
    with open(args.split_test, 'r') as testf:
        test_list = json.load(testf)

    #split dataset
    SplitFilesDCMPP('Train', train_list, args, split_dict, True)
    SplitFilesDCMPP('Validation', validation_list, args, split_dict, False)
    SplitFilesDCMPP('Test', test_list, args, split_dict, False)
    print('spliting file finished!')
    #extract music features
    aist_extract(f'{args.output_dir}/Train/Audio',f'{args.output_dir}/Train/aist_feats_long')
    aist_extract(f'{args.output_dir}/Validation/Audio',f'{args.output_dir}/Validation/aist_feats_long')
    aist_extract(f'{args.output_dir}/Test/Audio',f'{args.output_dir}/Test/aist_feats_long')
    print('extract music features finished!')
    # detect bone mask
    DetactBoneMask(f'{args.output_dir}/Train/CameraCentric',f'{args.output_dir}/Train/Simplified_MotionGlobalTransform',f'{args.output_dir}/Train/BoneMask')
    DetactBoneMask(f'{args.output_dir}/Validation/CameraCentric',f'{args.output_dir}/Validation/Simplified_MotionGlobalTransform',f'{args.output_dir}/Validation/BoneMask')
    DetactBoneMask(f'{args.output_dir}/Test/CameraCentric',f'{args.output_dir}/Test/Simplified_MotionGlobalTransform',f'{args.output_dir}/Test/BoneMask')
    print('detect bone mask finished!')
    GlobalTransform2Keypoints_File(f'{args.output_dir}/Train/Simplified_MotionGlobalTransform',f'{args.output_dir}/Train/Keypoints3D')
    GlobalTransform2Keypoints_File(f'{args.output_dir}/Validation/Simplified_MotionGlobalTransform',f'{args.output_dir}/Validation/Keypoints3D')
    GlobalTransform2Keypoints_File(f'{args.output_dir}/Test/Simplified_MotionGlobalTransform',f'{args.output_dir}/Test/Keypoints3D')
    print('get 3D keypoints from global transform finished!')

def WriteFileDCMPP(args, set_tag, sign_category, id_raw, start_keyframe, end_keyframe, sub_tag = None):
    # if sub_tag is not None:
    #     print(id_raw + "_" + sub_tag)
    # else:
    #     print(id_raw)
    if sub_tag is not None:
        cur_prefix = id_raw + "_" + sub_tag
    else:
        cur_prefix = id_raw

    aligned_audio_path = f'{args.audio_dir}/a{id_raw}.wav'
    aligned_camera_c_path = f'{args.camera_c_dir}/c{id_raw}.json'
    aligned_camera_kf_path = f'{args.camera_kf_dir}/c{id_raw}.json'
    aligned_motion_path = f'{args.motion_dir}/m{id_raw}_gt.json'

    out_audio_path = f'{args.output_dir}/{set_tag}/Audio/a{cur_prefix}_{sign_category}.wav'
    out_camera_c_path = f'{args.output_dir}/{set_tag}/CameraCentric/c{cur_prefix}.json'
    out_camera_kf_path = f'{args.output_dir}/{set_tag}/CameraKeyframe/c{cur_prefix}.json'
    out_motion_path = f'{args.output_dir}/{set_tag}/Simplified_MotionGlobalTransform/m{cur_prefix}_gt.json'

    aligned_audio, sr = lr.load(aligned_audio_path, sr=None)
    with open(aligned_camera_c_path, 'r') as ccf:
        aligned_camera_c_data = json.load(ccf)
    with open(aligned_camera_kf_path, 'r') as ckff:
        aligned_camera_kf_data = json.load(ckff)
    with open(aligned_motion_path, 'r') as mf:
        aligned_motion_data = json.load(mf)

    if end_keyframe == -1: # no split data
        tmp_audio = aligned_audio
    elif math.ceil(float(end_keyframe+1)/float(fps)*float(sr)) >= len(aligned_audio):
        tmp_audio = aligned_audio[int(float(start_keyframe)/float(fps)*float(sr)):]
    else:
        tmp_audio = aligned_audio[int(float(start_keyframe)/float(fps)*float(sr)):math.ceil(float(end_keyframe+1)/float(fps)*float(sr))]
    sf.write(out_audio_path, tmp_audio, sr)

    tmp_motion = aligned_motion_data.copy()
    tmp_camera_c = aligned_camera_c_data.copy()
    tmp_camera_kf = aligned_camera_kf_data.copy()
    if not end_keyframe == -1:# actually here is a small mistake because in this file there are all bone frames but not only bone keyframes, we keep this unchange to align with DanceCamera3D
        tmp_motion["BoneKeyFrameNumber"] = end_keyframe - start_keyframe + 1
        tmp_motion["BoneKeyFrameTransformRecord"] = aligned_motion_data["BoneKeyFrameTransformRecord"][start_keyframe:end_keyframe+1]
        if not tmp_motion["BoneKeyFrameNumber"] == len(tmp_motion["BoneKeyFrameTransformRecord"]):
            print(set_tag,cur_prefix,tmp_motion["BoneKeyFrameNumber"],len(tmp_motion["BoneKeyFrameTransformRecord"]))
    
    camera_keyframes = tmp_camera_kf["CameraKeyFrameRecord"]
    camera_keyframes.sort(key=FrameTimeFunc)
    out_keyframes_number = 0
    out_camera_keyframes = []
    
    if end_keyframe == -1:
        out_camera_keyframes = camera_keyframes.copy()
        out_keyframes_number = len(out_camera_keyframes)
        out_frame_time = out_camera_keyframes[-1]["FrameTime"] + 1
    else:
        for ckf in camera_keyframes:
            if ckf["FrameTime"] < start_keyframe:
                continue
            elif  ckf["FrameTime"] > end_keyframe:
                break
            else:
                out_camera_keyframes.append(ckf)
                out_keyframes_number += 1
        out_frame_time = end_keyframe - start_keyframe + 1
    with open(out_camera_kf_path, 'w') as ckfof:
        json.dump({
            "CameraKeyFrameRecord": out_camera_keyframes,
            "FrameTime": out_frame_time,
            "StartKeyframe": start_keyframe,
            "EndKeyframe": out_camera_keyframes[-1]["FrameTime"]
        },
                        ckfof,
                        indent=2,  
                        sort_keys=True,  
                        ensure_ascii=False)

    if not end_keyframe == -1:
        for camera_args in tmp_camera_c:
            if camera_args == 'FrameTime':
                tmp_camera_c[camera_args] = end_keyframe - start_keyframe + 1
            else:
                tmp_camera_c[camera_args] = tmp_camera_c[camera_args][start_keyframe:end_keyframe+1]

    # detect camera keyframe mask, 1 for keyframe 0 for not keyframe
    tmp_camera_c['KeyframeMask'] = np.zeros(len(tmp_camera_c["camera_eye"]))
    tmp_camera_c['KeyframePos'] = []
    # print(tmp_camera_kf.keys())
    for cur_kframe in out_camera_keyframes:
        if cur_kframe['FrameTime'] - start_keyframe < len(tmp_camera_c["camera_eye"]):
            tmp_camera_c['KeyframeMask'][cur_kframe['FrameTime'] - start_keyframe] = 1
            tmp_camera_c['KeyframePos'].append(cur_kframe['FrameTime'] - start_keyframe)
        else:
            continue
    tmp_camera_c['KeyframeMask'] = tmp_camera_c['KeyframeMask'].tolist()
    with open(out_camera_c_path, 'w') as ccof:
        json.dump(tmp_camera_c,
                ccof,
                indent=2,  
                sort_keys=True,  
                ensure_ascii=False)
    with open(out_motion_path, 'w') as mof:
        json.dump(tmp_motion,
                mof,
                indent=2,  
                sort_keys=True,  
                ensure_ascii=False)

    





def SplitFilesDCMPP(set_tag, id_list, args, split_dict, merge_adjacent):
    
    print('start spliting files for ',set_tag,' set……')
    Path(f'{args.output_dir}/{set_tag}/Audio').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/{set_tag}/CameraCentric').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/{set_tag}/CameraKeyframe').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/{set_tag}/Simplified_MotionGlobalTransform').mkdir(parents=True, exist_ok=True)

    if merge_adjacent:# we merge the adjacent sub-sequences in train data only to preserve more keyframe context
        merged_id_list = []
        id_list = sorted(id_list, key=GetSplitIndex)
        id_sum = len(id_list)

        id_spacer_list = [0]
        for id_i in range(1, id_sum):
            cur_id_split = id_list[id_i].split('_')
            pre_id_split = id_list[id_i-1].split('_')
            if len(cur_id_split) == 2 or len(pre_id_split) == 2:
                id_spacer_list.append(id_i)
            elif cur_id_split[1] == pre_id_split[1] and int(cur_id_split[2]) == int(pre_id_split[2]) + 1:
                continue
            else:
                id_spacer_list.append(id_i)
        id_spacer_list.append(id_sum)
        for spacer_i in tqdm(range(1,len(id_spacer_list))):
            if id_spacer_list[spacer_i] == id_spacer_list[spacer_i-1] + 1: # one piece of data
                cur_id_raw = id_list[id_spacer_list[spacer_i-1]].split("_")[1]
                sign_category = id_list[id_spacer_list[spacer_i-1]][0]
                if len(split_dict[cur_id_raw]) == 0: # raw data no split
                    start_keyframe = 0
                    end_keyframe = -1
                    WriteFileDCMPP(args, set_tag, sign_category, cur_id_raw, start_keyframe, end_keyframe) # for not split data, align with DCM
                else: # split data
                    cur_id_sub = id_list[id_spacer_list[spacer_i-1]].split("_")[2]
                    start_keyframe = split_dict[cur_id_raw][int(cur_id_sub)][0]
                    end_keyframe = split_dict[cur_id_raw][int(cur_id_sub)][1]
                    WriteFileDCMPP(args, set_tag, sign_category, cur_id_raw, start_keyframe, end_keyframe, sub_tag = cur_id_sub)
            else: # not one piece of data
                cur_id_raw = id_list[id_spacer_list[spacer_i-1]].split("_")[1]
                sign_category = id_list[id_spacer_list[spacer_i-1]][0]
                cur_id_sub_start = id_list[id_spacer_list[spacer_i-1]].split("_")[2]
                cur_id_sub_end = id_list[id_spacer_list[spacer_i] - 1].split("_")[2]
                start_keyframe = split_dict[cur_id_raw][int(cur_id_sub_start)][0]
                end_keyframe = split_dict[cur_id_raw][int(cur_id_sub_end)][1]
                WriteFileDCMPP(args, set_tag, sign_category, cur_id_raw, start_keyframe, end_keyframe, sub_tag = cur_id_sub_start+"~"+cur_id_sub_end)
                
    else:# we keep the test or validation set the same as that in DanceCamera3D
        for index_i in tqdm(id_list):
            cur_id_raw = index_i.split("_")[1]
            sign_category = index_i[0]
            if len(split_dict[cur_id_raw]) == 0: # raw data
                start_keyframe = 0
                end_keyframe = -1
                WriteFileDCMPP(args, set_tag, sign_category, cur_id_raw, start_keyframe, end_keyframe)
            else:
                cur_id_sub = index_i.split("_")[2]
                start_keyframe = split_dict[cur_id_raw][int(cur_id_sub)][0]
                end_keyframe = split_dict[cur_id_raw][int(cur_id_sub)][1]
                WriteFileDCMPP(args, set_tag, sign_category, cur_id_raw, start_keyframe, end_keyframe, sub_tag = cur_id_sub)



parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', type=str, required=True)
parser.add_argument('--camera_kf_dir', type=str, required=True)
parser.add_argument('--camera_c_dir', type=str, required=True)
parser.add_argument('--motion_dir', type=str, required=True)
parser.add_argument('--split_train', type=str, required=True)
parser.add_argument('--split_validation', type=str, required=True)
parser.add_argument('--split_test', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--split_record', type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    
    MakeDCMPlusPlus(args)
    