python3 scripts/make_dcm_plusplus.py \
--audio_dir DCM_data/amc_aligned_data/Audio \
--camera_kf_dir DCM_data/amc_aligned_data/CameraKeyframe \
--camera_c_dir DCM_data/amc_aligned_data/CameraCentric \
--motion_dir DCM_data/amc_aligned_data/Simplified_MotionGlobalTransform/ \
--split_train DCM_data/split/train.json \
--split_validation DCM_data/split/validation.json \
--split_test DCM_data/split/test.json \
--output_dir DCM++ \
--split_record DCM_data/split/long2short.json
