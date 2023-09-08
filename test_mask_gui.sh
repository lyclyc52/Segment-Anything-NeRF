
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/data/garden \
--enable_cam_center \
--with_mask \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_rgb/checkpoints/ngp.pth \
--H 512 \
--test \
--gui \
--data_type mip \
--num_rays 16384 \
--render_mask_instance 1 \
--render_mask_type composition \
--mask_mlp_type default \
--ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_garden_mask_1_default/checkpoints/ngp_ep0011.pth
# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_with_rgb_sim_6_adaptive_large_patch/checkpoints/ngp_ep0012.pth
# 
 



