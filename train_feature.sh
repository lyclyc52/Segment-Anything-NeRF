export CUDA_VISIBLE_DEVICES=2
python main.py /ssddata/yliugu/data/Datasets/garden \
--workspace trial_model/trial_garden_sam_sumafter \
--enable_cam_center \
--with_sam \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_garden_rgb/checkpoints/ngp_ep0088.pth \
--data_type mip \
--iters 5000 \
--contract \
--sum_after_mlp