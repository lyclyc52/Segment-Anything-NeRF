
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 
export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/teatime \
--workspace trial2_teatime \
--enable_cam_center \
--with_sam \
--init_ckpt /ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/checkpoints/ngp.pth \
--H 384 \
--test \
--gui \
--data_type lerf 
