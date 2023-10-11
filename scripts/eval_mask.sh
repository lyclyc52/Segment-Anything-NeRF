
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=0
python main.py /ssddata/yliugu/data/kitchen \
--workspace /ssddata/yliugu/Segment-Anything-NeRF/trial_model/mask_auto_default/ \
--enable_cam_center \
--with_mask \
--H 512 \
--test \
--data_type mip \
--render_mask_type composition \
--sum_after_mlp \
--mask_mlp_type adaptive \
--adaptive_mlp_type density \
--render_mask_instance_id -1 \
--n_inst 100 \
--val_all