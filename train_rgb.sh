export CUDA_VISIBLE_DEVICES=2
python main.py /ssddata/yliugu/data/garden \
--workspace trial_garden_rgb \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--iters 30000 \
--contract