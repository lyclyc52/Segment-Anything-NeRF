export CUDA_VISIBLE_DEVICES=0
python main.py /ssddata/yliugu/teatime \
--workspace mask_teatime \
--enable_cam_center \
--with_mask \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial1_teatime/checkpoints/ngp_ep0065.pth \
--data_type mip \
--patch_size 64 \
--num_rays 4096 \
--iters 5000 \
--mask_folder_name masks \
--use_wandb

