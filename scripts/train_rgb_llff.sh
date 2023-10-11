export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/data/tincan \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/rgb_nerf/tincan \
--enable_cam_center \
--downscale 1 \
--data_type llff \
--iters 10000 \
--contract \
--random_image_batch \
--min_near 0.2 