python3 scripts/test_stage2n3.py \
--camera_format polar \
--use_generate_keypoints \
--generated_keyframemask_dir output/stage1/test_stage1_test/test \
--checkpoint checkpoints/DCA_stage2n3.pt \
--render_dir output \
--processed_data_dir DCM++/mix_dataset_backups/DCA \
--force_reload \
--exp_name DCA

# results can be found in output/test_DCA/etest
# add --render_videos to start rendering during the inference process