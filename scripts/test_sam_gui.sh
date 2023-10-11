
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/teatime \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/sam_nerf/ctr_lift_0 \
--enable_cam_center \
--with_sam \
--H 512 \
--test \
--gui \
--data_type mip \
--num_rays 16384 \
--contract \
--sam_use_view_direction