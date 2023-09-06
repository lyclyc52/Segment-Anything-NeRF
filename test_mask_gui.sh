
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/teatime \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/sam_mask_teatime \
--enable_cam_center \
--with_mask \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial1_teatime/checkpoints/ngp.pth \
--H 512 \
--test \
--gui \
--data_type mip \
--num_rays 16384 \
--render_mask_instance 1 \
--render_mask_type composition \
--ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_with_rgb_sim_3/checkpoints/ngp_ep0031.pth \
# --redundant_instance 10
# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_with_rgb_sim_6_contr/checkpoints/ngp_ep0016.pth \
# --redundant_instance 20
# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_with_rgb_sim_4_new_loss/checkpoints/ngp_ep0016.pth
# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_origin_select_2/checkpoints/ngp_ep0004.pth
# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_origin_select_3/checkpoints/ngp_ep0016.pth
