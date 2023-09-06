export CUDA_VISIBLE_DEVICES=2
python main.py /ssddata/yliugu/teatime \
--workspace trial_teatime_sam \
--enable_cam_center \
--with_sam \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_teatime_rgb/checkpoints/ngp_ep0091.pth \
--data_type mip \
--iters 5000 \
--contract