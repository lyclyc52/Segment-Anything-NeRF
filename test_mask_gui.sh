
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/teatime \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/sam_mask_teatime \
--enable_cam_center \
--with_mask \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial1_teatime/checkpoints/ngp.pth \
--H 512 \
--test \
--gui \
--data_type mip \
--num_rays 8192 \
--render_mask_instance 1 \
--render_mask_type composition \
--ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_teatime/checkpoints/ngp_ep0010.pth
