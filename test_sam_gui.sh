
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/teatime \
--workspace trial_teatime_sam \
--enable_cam_center \
--with_sam \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_teatime_rgb/checkpoints/ngp_ep0091.pth \
--H 512 \
--test \
--gui \
--data_type mip \
--num_rays 16384 \
--contract
