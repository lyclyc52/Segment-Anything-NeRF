
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
--render_mask_instance 1 \
--render_mask_type composition \
--sum_after_mlp \
--mask_mlp_type adaptive \
--ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/table_vase/checkpoints/ngp_ep0005.pth \
--adaptive_mlp_type density 



# --ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_mask_with_rgb_sim_6_adaptive_large_patch/checkpoints/ngp_ep0012.pth
# 
 



