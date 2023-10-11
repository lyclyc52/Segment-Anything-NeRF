export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/data/ai_001_008 \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/rgb_nerf/ai_001_008 \
--enable_cam_center \
--downscale 1 \
--data_type lift \
--test \
--test_split val \
--val_type val_all \
--num_rays 16384 \
--contract \
--sam_use_view_direction