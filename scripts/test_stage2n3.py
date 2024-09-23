import glob
import os
import sys
from pathlib import Path
import numpy as np
from args import parse_test_opt
from torch.utils.data import DataLoader
from EditableDanceCameraVelocity import EditableDanceCameraVelocity
import pickle
from dataset.dcmpp_dataset import DCMPPDataset

def test_stage2_use_generate_keypoints(opt):
    model = EditableDanceCameraVelocity(feature_type = opt.feature_type,
                            checkpoint_path = opt.checkpoint,
                            EMA_tag = False
                            )
    model.eval()

    normalizer_pose = model.normalizer_pose
    normalizer_camera_dis = model.normalizer_camera_dis
    normalizer_camera_pos = model.normalizer_camera_pos
    normalizer_camera_rot = model.normalizer_camera_rot
    normalizer_camera_fov = model.normalizer_camera_fov
    normalizer_camera_eye = model.normalizer_camera_eye

    test_dataset = DCMPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer_pose = model.normalizer_pose,
                normalizer_camera_dis = model.normalizer_camera_dis,
                normalizer_camera_pos = model.normalizer_camera_pos,
                normalizer_camera_rot = model.normalizer_camera_rot,
                normalizer_camera_fov = model.normalizer_camera_fov,
                normalizer_camera_eye = model.normalizer_camera_eye,
                force_reload=opt.force_reload,
                history_len = model.history_len,
                inference_len = model.inference_len,
                feature_type = opt.feature_type,
                loaded_normalizer = True,
                generated_keyframemask_dir = opt.generated_keyframemask_dir
            )
    print(test_dataset.subsequence_end_index)
    test_data_loader = DataLoader(
            test_dataset,
            batch_size=test_dataset.subsequence_end_index[-1],
            # shuffle=True,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    print("Generating dances")
    (x_camera, x_camera_imask, x_bone_mask, pose_cond, music_cond, x_wavpath, x_pre_padding, x_suf_padding, x_start_frame, x_end_frame) = next(iter(test_data_loader))
    pose_cond = pose_cond.to(model.accelerator.device)
    music_cond = music_cond.to(model.accelerator.device)
    # render_count = len(test_dataset.subsequence_end_index)
    sample_batch_size = test_dataset.subsequence_end_index[-1]

    

    model.render_sample(
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
        normalizer_pose,
        normalizer_camera_dis,
        normalizer_camera_pos,
        normalizer_camera_rot,
        normalizer_camera_fov,
        normalizer_camera_eye,
        'test',
        os.path.join(opt.render_dir, "test_" + opt.exp_name),
        render_videos=opt.render_videos,
        sound=True,
        is_train=False,
        inserted_keyframe = test_dataset.inserted_keyframe,
    )
    print("Done")

def test_stage2_use_gt_keypoints(opt):
    print(opt.feature_type)
    model = EditableDanceCameraVelocity(feature_type = opt.feature_type,
                            checkpoint_path = opt.checkpoint,
                            EMA_tag = False
                            )
    model.eval()

    _, test_dataset = model.load_datasets(opt)

    normalizer_pose = model.normalizer_pose
    normalizer_camera_dis = model.normalizer_camera_dis
    normalizer_camera_pos = model.normalizer_camera_pos
    normalizer_camera_rot = model.normalizer_camera_rot
    normalizer_camera_fov = model.normalizer_camera_fov
    normalizer_camera_eye = model.normalizer_camera_eye

    test_data_loader = DataLoader(
            test_dataset,
            batch_size=test_dataset.subsequence_end_index[-1],
            # shuffle=True,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    print("Generating dances")
    (x_camera, x_camera_imask, x_bone_mask, pose_cond, music_cond, x_wavpath, x_pre_padding, x_suf_padding, x_start_frame, x_end_frame) = next(iter(test_data_loader))
    pose_cond = pose_cond.to(model.accelerator.device)
    music_cond = music_cond.to(model.accelerator.device)

    sample_batch_size = test_dataset.subsequence_end_index[-1]
    print(sample_batch_size)

    

    model.render_sample(
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
        normalizer_pose,
        normalizer_camera_dis,
        normalizer_camera_pos,
        normalizer_camera_rot,
        normalizer_camera_fov,
        normalizer_camera_eye,
        'test',
        os.path.join(opt.render_dir, "test_" + opt.exp_name),
        render_videos=opt.render_videos,
        sound=True,
        is_train=False,
        use_gt_keyframe_parameters = opt.use_gt_keyframe_parameters
    )

    print("Done")
if __name__ == "__main__":
    opt = parse_test_opt()
    if opt.use_generate_keypoints:
        test_stage2_use_generate_keypoints(opt)
    else:
        test_stage2_use_gt_keypoints(opt)