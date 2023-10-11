export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python main.py /ssddata/yliugu/data/kitchen \
--enable_cam_center \
--with_mask \
--data_type mip \
--patch_size 1 \
--num_rays 6000 \
--iters 1000 \
--contract \
--rgb_similarity_loss_weight 5 \
--label_regularization_weight 0 \
--rgb_similarity_threshold 0.1 \
--incoherent_uncertainty_weight 1 \
--redundant_instance 0 \
--n_inst 100 \
--mask_mlp_type default \
--adaptive_mlp_type density \
--mask_folder_name auto_masks \
--rgb_similarity_num_sample 20 \
--label_regularization_weight 0 \
--sum_after_mlp \
--workspace trial_model/mask_auto_default \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_kitchen_rgb/checkpoints/ngp_ep0090.pth \
--rgb_similarity_iter 300 \
--preload \
--num_local_sample 64 \
--local_sample_patch_size 16 \
--mixed_sampling \
--incoherent_update_iter 50 \
--use_dynamic_incoherent \
--incoherent_downsample_scale 4 \
--error_map \
# --use_multi_res
# --random_image_batch \

# --rgb_similarity_use_pred_logistics \