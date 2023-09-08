export CUDA_VISIBLE_DEVICES=2
python main.py /ssddata/yliugu/data/garden \
--workspace trial_garden_sam \
--enable_cam_center \
--with_sam \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_garden_rgb/checkpoints/ngp_ep0088.pth \
--data_type mip \
--iters 5000 \
--contract