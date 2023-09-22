export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/data/shoe_rack \
--workspace trial_model/trial_shoe_rack_rgb \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--iters 15000 \
--contract \
--random_image_batch