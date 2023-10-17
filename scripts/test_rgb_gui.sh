
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/data/horns \
--workspace /ssddata/yliugu/trial_model_final/rgb_nerf/garden \
--enable_cam_center \
--H 512 \
--test \
--gui \
--data_type lift \
--contract \
--sum_after_mlp \
--max_ray_batch 65536 \
