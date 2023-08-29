export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/teatime \
--workspace trial2_teatime \
--enable_cam_center \
--with_sam \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial1_teatime/checkpoints/ngp_ep0065.pth \
--data_type mip \
--iters 5000
