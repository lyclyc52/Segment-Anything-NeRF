export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
python main.py /ssddata/yliugu/data/Datasets/garden \
--workspace trial_model/debug_origin_gt \
--enable_cam_center \
--with_mask \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_rgb/checkpoints/ngp_ep0088.pth \
--data_type mip \
--patch_size 1 \
--num_rays 4096 \
--iters 5000 \
--mask_folder_name table_gt \
--rgb_similarity_loss_weight 0 \
--rgb_similarity_threshold 0.1 \
--rgb_similarity_num_sample 100 \
--label_regularization_weight 0 \
--incoherent_uncertainty_weight 1 \
--contract \
--mask_mlp_type default \
--random_image_batch




