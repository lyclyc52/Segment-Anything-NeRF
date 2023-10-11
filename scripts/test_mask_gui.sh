
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/data/garden \
--enable_cam_center \
--with_mask \
--H 512 \
--test \
--gui \
--data_type mip \
--render_mask_type composition \
--sum_after_mlp \
--mask_mlp_type default \
--ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/mask_blue_shoes_default/checkpoints/ngp_ep0015.pth \
--adaptive_mlp_type density \
--render_mask_instance_id 1 \
--n_inst 2 \



# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_with_rgb_sim_6_adaptive_large_patch/checkpoints/ngp_ep0012.pth
# 
 



