export CUDA_VISIBLE_DEVICES=1
python main.py /ssddata/yliugu/data/shoe_rack \
--workspace trial_model/trial_shoe_rack_sam \
--enable_cam_center \
--with_sam \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial_model/trial_shoe_rack_rgb/checkpoints/ngp_ep0120.pth \
--data_type mip \
--iters 5000 \
--contract \
--sam_use_view_direction