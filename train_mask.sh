export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
python main.py /ssddata/yliugu/data/garden \
--workspace trial_model/trial_garden_mask_2_adaptive \
--enable_cam_center \
--with_mask \
--data_type mip \
--patch_size 64 \
--num_rays 4096 \
--iters 5000 \
--contract \
--rgb_similarity_loss_weight 1 \
--rgb_similarity_threshold 0.2 \
--incoherent_uncertainty_weight 1 \
--redundant_instance 0 \
--mask_mlp_type adaptive \
--mask_folder_name table \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_rgb/checkpoints/ngp_ep0088.pth


