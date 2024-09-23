import glob
import os
import sys
from pathlib import Path
import numpy as np
from args import parse_test_stage1_opt
from torch.utils.data import DataLoader
from MusicDance2Keyframe import MusicDance2Keyframe
import pickle
from dataset.dckm_dataset import DCKMDataset

def test_stage1(opt):
    model = MusicDance2Keyframe(feature_type = opt.feature_type,
                        checkpoint_path = opt.checkpoint,
                        EMA_tag = False
                        )
    model.eval()


    # load datasets
    _, test_dataset = model.load_datasets(opt)

    test_data_loader = DataLoader(
            test_dataset,
            batch_size=test_dataset.subsequence_end_index[-1],#process all the data within one batch
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

    print("Generating dances")
    (x_camera_kf, x_padding_mask, pose_cond, music_cond, x_wavpath) = next(iter(test_data_loader))
    pose_cond = pose_cond.to(model.accelerator.device)
    music_cond = music_cond.to(model.accelerator.device)
    sample_batch_size = test_dataset.subsequence_end_index[-1]

    model.render_sample(
            sample_batch_size,
            pose_cond,
            music_cond,
            x_camera_kf,
            x_padding_mask,
            x_wavpath,
            'test',
            os.path.join(opt.save_dir, "test_" + opt.exp_name),
            is_train=False
        )
        
if __name__ == "__main__":
    opt = parse_test_stage1_opt()
    test_stage1(opt)