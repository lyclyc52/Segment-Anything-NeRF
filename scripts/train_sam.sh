export CUDA_VISIBLE_DEVICES=2
python main.py /ssddata/yliugu/data/kitchen \
--workspace /ssddata/yliugu/trial_model_final/mask_nerf/garden-table-nerf \
--enable_cam_center \
--with_sam \
--data_type mip \
--iters 5000 \
--contract \
--sam_use_view_direction