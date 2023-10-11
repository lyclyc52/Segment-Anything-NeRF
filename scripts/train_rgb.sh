export CUDA_VISIBLE_DEVICES=0
python main.py /ssddata/yliugu/data/ctr_lift_2 \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/rgb_nerf/ctr_lift_2 \
--enable_cam_center \
--downscale 1 \
--data_type lift \
--iters 200000 \
--contract \
--val_type default \
--random_image_batch 