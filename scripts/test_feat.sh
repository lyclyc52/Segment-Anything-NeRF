export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/data/bonsai \
--workspace trial_model/bonsai \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--test \
--test_split test \
--with_sam \
--num_rays 16384 \
--contract \
--sam_use_view_direction \
--return_extra