python3 scripts/test_stage2n3.py \
--camera_format polar \
--checkpoint checkpoints/DCA_stage2n3.pt \
--render_dir output \
--processed_data_dir DCM++/dataset_backups \
--force_reload \
--use_gt_keyframe_parameters \
--exp_name DCA_ablate_stage2n3_w_gt

# results can be found in output/test_DCA_ablate_stage2n3_w_gt/etest
# add --render_videos to start rendering during the inference process