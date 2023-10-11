export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/data/shoe_rack \
--workspace trial_model/trial_shoe_rack_sam \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--test \
--test_split val \
--with_sam \
--val_all \
--num_rays 16384 \
--contract \
--sam_use_view_direction