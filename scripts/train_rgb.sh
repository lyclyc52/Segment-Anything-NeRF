export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/data/kitchen \
--workspace /ssddata/yliugu/trial_model_final/rgb_nerf/kitchen \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--iters 20000 \
--contract \
--val_type default \
--random_image_batch 