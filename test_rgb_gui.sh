
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=2
python main.py /ssddata/yliugu/teatime \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_teatime_rgb \
--enable_cam_center \
--H 512 \
--test \
--gui \
--data_type mip \
# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_origin_select_2/checkpoints/ngp_ep0004.pth
