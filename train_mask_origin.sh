export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
python main.py /ssddata/yliugu/teatime \
--workspace trial_mask_origin_select_8_default \
--enable_cam_center \
--with_mask \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_teatime_rgb/checkpoints/ngp_ep0091.pth \
--data_type mip \
--patch_size 64 \
--num_rays 4096 \
--iters 5000 \
--mask_folder_name sheep_masks_without_negative \
--rgb_similarity_loss_weight 0 \
--rgb_similarity_threshold 0.3 \
--incoherent_uncertainty_weight 1 \
--contract \
--mask_mlp_type default

