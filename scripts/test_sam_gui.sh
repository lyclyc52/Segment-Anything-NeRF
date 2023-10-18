
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/teatime \
--workspace /ssddata/yliugu/trial_model_final/sam_nerf/waldo_kitchen \
--enable_cam_center \
--with_sam \
--H 512 \
--test \
--gui \
--data_type mip \
--max_ray_batch 32768 \
--contract \
--sam_use_view_direction