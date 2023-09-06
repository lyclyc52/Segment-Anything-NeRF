export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
python main.py /ssddata/yliugu/teatime \
--workspace trial_mask_with_rgb_sim_3 \
--enable_cam_center \
--with_mask \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_teatime_rgb/checkpoints/ngp_ep0091.pth \
--data_type mip \
--patch_size 100 \
--num_rays 8192 \
--iters 10000 \
--contract \
--mask_folder_name sheep_masks \
--rgb_similarity_loss_weight 1 \
--rgb_similarity_threshold 0.2 \
--incoherent_uncertainty_weight 1 \
--redundant_instance 0 \


