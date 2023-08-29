export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/teatime \
--workspace trial2_teatime \
--enable_cam_center \
--downscale 4 \
--data_type mip \
--test \
--test_split val \
--with_sam