export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/data/3dfront_0091_00 \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/rgb_nerf/3dfront \
--enable_cam_center \
--downscale 1 \
--data_type 3dfront \
--iters 1000 \
--contract \
--random_image_batch \
--min_near 0.2 