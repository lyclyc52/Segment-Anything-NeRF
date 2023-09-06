export CUDA_VISIBLE_DEVICES=2
python main.py /ssddata/yliugu/teatime \
--workspace trial_teatime_sam \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--test \
--test_split val \
--with_sam \
--val_all