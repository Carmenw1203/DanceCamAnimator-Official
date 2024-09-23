import argparse


def parse_train_stage1_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs_stage1/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="DCM++", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="DCM++/stage1_dataset_backups/",
        help="Dataset backup path",
    )

    parser.add_argument(
        "--render_dir", type=str, default="renders_stage1/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="aist")# jukebox or aist
    parser.add_argument(
        "--wandb_pj_name", type=str, default="MusicDance2Keyframe_S1", help="project name"
    )

    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
    parser.add_argument(
        "--w_positive_loss",
        type=float,
        default=3.0,
        help="loss weight",
    )
    parser.add_argument(
        "--w_negative_loss",
        type=float,
        default=0.5,
        help="velocity loss weight",
    )
    opt = parser.parse_args()
    return opt
# for stage 2
def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--backbone",default="diffusion", help="backbone")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="DCM++", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )

    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument(
        "--render_videos", action="store_true", help="whether to render videos"
    )

    parser.add_argument("--feature_type", type=str, default="aist")# jukebox or aist
    parser.add_argument(
        "--wandb_pj_name", type=str, default="EditableDanceCamera_S2", help="project name"
    )

    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
    parser.add_argument(
        "--camera_format",
        type=str,
        default="centric",# or centric
        help="polar coordinates or centric",
    )
    parser.add_argument(
        "--w_loss",
        type=float,
        default=2,
        help="loss weight",
    )
    parser.add_argument(
        "--w_v_loss",
        type=float,
        default=5,
        help="velocity loss weight",
    )
    parser.add_argument(
        "--w_a_loss",
        type=float,
        default=5,
        help="acceleration loss weight",
    )
    parser.add_argument(
        "--w_in_ba_loss",
        type=float,
        default=0.0015,
        help="inside body parts attention loss weight",
    )
    parser.add_argument(
        "--w_out_ba_loss",
        type=float,
        default=0,
        help="outside body parts attention loss weight",
    )
    opt = parser.parse_args()
    return opt

def parse_test_stage1_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="aist")# jukebox or aist
    parser.add_argument(
        "--checkpoint", type=str, default="train-3000.pt", help="checkpoint"
    )
    parser.add_argument(
        "--save_dir", type=str, default="keypoint_save/", help="Sample save path"
    )

    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="DCM++/stage1_dataset_backups",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument("--data_path", type=str, default="DCM++", help="raw data path")
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    opt = parser.parse_args()
    return opt

def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="aist")# jukebox or aist
    parser.add_argument("--data_path", type=str, default="DCM++", help="raw data path")
    parser.add_argument(
        "--checkpoint", type=str, default="train-3000.pt", help="checkpoint"
    )
    parser.add_argument(
        "--use_generate_keypoints", action="store_true", help="whether to use generated keypoints"
    )
    parser.add_argument(
        "--use_gt_keyframe_parameters", action="store_true", help="whether to use generated keypoints"
    )
    parser.add_argument(
        "--generated_keyframemask_dir", type=str, default="", help="directory of generated keyframe mask"
    )
    
    parser.add_argument(
        "--camera_format",
        type=str,
        default="centric",# or centric
        help="polar coordinates or centric",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument(
        "--render_videos", action="store_true", help="whether to render videos"
    )

    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    opt = parser.parse_args()
    return opt