export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python main.py /ssddata/yliugu/data/Datasets/garden \
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
--mask_folder_name table_sumafter \
--sum_after_mlp \
--rgb_similarity_num_sample 100 \
--label_regularization_weight 1 \
--adaptive_mlp_type rgb \
--workspace trial_model/debug \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_rgb/checkpoints/ngp_ep0088.pth \

