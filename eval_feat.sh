export CUDA_VISIBLE_DEVICES=3
python main.py /ssddata/yliugu/data/Datasets/garden \
--workspace trial_model/trial_garden_sam_sumafter \
--enable_cam_center \
--downscale 1 \
--data_type mip \
--test \
--test_split val \
--with_sam \
--val_all \
--sum_after_mlp