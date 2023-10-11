
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/data/horns \
--enable_cam_center \
--H 512 \
--test \
--gui \
--data_type lift \
--contract \
--ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/rgb_nerf/ctr_lift_2/checkpoints/ngp.pth \
--sum_after_mlp \
