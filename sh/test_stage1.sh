python3 scripts/test_stage1.py \
--checkpoint checkpoints/DCA_stage1.pt \
--save_dir output/stage1 \
--processed_data_dir DCM++/stage1_dataset_backups \
--force_reload \
--exp_name stage1_test

# results can be found in output/stage1/test_stage1_test/test