import numpy as np
import os
import os.path as path
import json
from tqdm import tqdm
import shutil
import glob
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

scene_path = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
metadata_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_metadata_v2.json'
workspace_root = '/ssddata/yliugu/trial_model_final/sam_nerf'
data_root = '/ssddata/yliugu/data'
with open(scene_path) as f:
    scene_dict = json.load(f)
    
scene_root = '/ssddata/yliugu/data'
scene_list = []
with open(metadata_path) as f:
    meta = json.load(f)

scene_list = list(meta.keys())
scene_list = ["donuts"]
for scene_name in scene_list:
    
    sam_path = path.join(workspace_root, scene_name)
    ckpt_path = path.join(sam_path, 'checkpoints')
    checkpoint_list = sorted(glob.glob(f'{ckpt_path}/*.pth'))
    if len(checkpoint_list) == 0:
        continue
    latest = path.basename(checkpoint_list[-1])
    epoch = latest[-8:-4]

    for object_name in list(meta[scene_name].keys()):
        
        
        object_name = 'donut_1'
        
        cmd = ['python sam_project.py ',
                '--files_root', workspace_root,
                '--output_root',  data_root,
                '--scene_name', scene_name,
                '--scene_object', object_name,
                '--epoch', epoch,
                '--meta_path', metadata_path,
                '--decode',
                '--purpose train'
            ]
        # cmd = ' '.join(cmd)
        choice_0 = ['--use_nerf_feature', '']
        for c_0 in choice_0:
            cur_cmd = cmd.copy()
            cur_cmd.append(c_0)
            cur_cmd = ' '.join(cur_cmd)
            os.system(cur_cmd)

        break
    break


# for scene_type in list(scene_dict.keys()):
#     for scene in tqdm(scene_dict[scene_type]):

        
#         cmd = ['python main.py ',
#                 path.join(scene_root, scene),
#                 '--workspace',
#                 path.join(workspace, 'sam_nerf', scene),
#                 '--enable_cam_center', 
#                 '--downscale 1',
#                 '--data_type',
#                 scene_type, 
#                 '--test', 
#                 '--test_split val', 
#                 "--val_type val_all",
#                 '--with_sam',
#                 '--num_rays 16384',
#                 '--contract', 
#                 '--sam_use_view_direction', 
#                 '--return_extra'
#             ]
#         cmd = ' '.join(cmd)
#         os.system(cmd)
