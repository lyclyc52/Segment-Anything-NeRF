export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/data/garden \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_sam \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--test \
--test_split val \
--with_sam \
--val_all