
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=0
python main.py /ssddata/yliugu/data/garden \
--workspace /ssddata/yliugu/trial_model_final/mask_nerf/donuts-donut_1-nerf \
--enable_cam_center \
--with_mask \
--H 512 \
--test \
--gui \
--data_type mip \
--render_mask_type composition \
--sum_after_mlp \
--mask_mlp_type default \
--adaptive_mlp_type density \
--render_mask_instance_id 1 \
--n_inst 2 \



# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_with_rgb_sim_6_adaptive_large_patch/checkpoints/ngp_ep0012.pth
# 
 



