export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python main.py /ssddata/yliugu/data/Datasets/garden \
--enable_cam_center \
--with_mask \
--data_type mip \
--patch_size 1 \
--num_rays 6000 \
--iters 5000 \
--contract \
--rgb_similarity_loss_weight 5 \
--rgb_similarity_threshold 0.1 \
--incoherent_uncertainty_weight 1 \
--redundant_instance 0 \
--mask_mlp_type adaptive \
--mask_folder_name table_sumafter \
--rgb_similarity_num_sample 50 \
--label_regularization_weight 0 \
--adaptive_mlp_type density \
--sum_after_mlp \
--workspace trial_model/debug_density \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_rgb/checkpoints/ngp_ep0088.pth \
--rgb_similarity_iter 210 \
--preload \
--random_image_batch \
--num_local_sample 8 \
--local_sample_patch_size 16 \
--mixed_sampling \
--use_dynamic_incoherent
