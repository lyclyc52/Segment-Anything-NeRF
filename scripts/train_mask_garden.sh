export CUDA_VISIBLE_DEVICES=3
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
--rgb_similarity_threshold 0.15 \
--incoherent_uncertainty_weight 1 \
--redundant_instance 0 \
--mask_mlp_type adaptive \
--mask_folder_name table_vase_sam \
--rgb_similarity_num_sample 20 \
--label_regularization_weight 0 \
--adaptive_mlp_type density \
--sum_after_mlp \
--workspace trial_model/table_vase \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_rgb/checkpoints/ngp_ep0088.pth \
--rgb_similarity_iter 200 \
--preload \
--num_local_sample 16 \
--local_sample_patch_size 32 \
--mixed_sampling \
--incoherent_update_iter 50 \
--use_dynamic_incoherent \
--incoherent_downsample_scale 4 \
--error_map \
--use_multi_res
# --random_image_batch \

# --rgb_similarity_use_pred_logistics \