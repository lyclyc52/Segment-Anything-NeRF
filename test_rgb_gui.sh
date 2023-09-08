
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=2
python main.py /ssddata/yliugu/data/garden \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden \
--enable_cam_center \
--H 512 \
--test \
--gui \
--data_type mip \
--contract \
--ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_rgb/checkpoints/ngp_ep0088.pth \
--sum_after_mlp
