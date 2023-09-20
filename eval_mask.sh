export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/data/Datasets/garden \
--workspace trial_model/garden_mask_adaptive_density_without_rgb_loss \
--ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/garden_mask_adaptive_density_without_rgb_loss/checkpoints/ngp_ep0001.pth \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--test \
--test_split val \
--with_mask \
--val_all \
--mask_folder_name table_sumafter \
--sum_after_mlp \
--num_rays 16384 \
--render_mask_instance 1 \
--mask_mlp_type adaptive \
--adaptive_mlp_type density 