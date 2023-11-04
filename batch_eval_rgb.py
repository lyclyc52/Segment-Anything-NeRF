import numpy as np
import os
import json
from tqdm import tqdm
import shutil
import os.path as path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

scene_dict = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
workspace = '/ssddata/yliugu/trial_model_final'
with open(scene_dict) as f:
    scene_dict = json.load(f)
    


data_root = '/ssddata/yliugu/data'
metadata_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_metadata_v2.json'
with open(metadata_path) as f:
    meta = json.load(f)
    
for scene_type in list(scene_dict.keys()):
    
    # scene_type = 'mip'
    scene_list = scene_dict[scene_type]
    
    for scene_name in scene_list:
        
        # scene_name = 'garden'
        scene_data_root = path.join(data_root, scene_name)
        
        for object_name in meta[scene_name]:
            ending = 'nerf'
            cur_scene_root = os.path.join(data_root, scene_name)
            mask_folder_name = f'train_{object_name}_{ending}'
            
            
            cmd = ['python main.py', cur_scene_root,
                    '--workspace', os.path.join(workspace, 'rgb_nerf', scene_name),
                    '--enable_cam_center', 
                    '--downscale 1',
                    '--data_type', scene_type, 
                    '--test', 
                    '--test_split val', 
                    '--val_type val_all',
                    '--num_rays 16384',
                    '--contract', 
                    '--sam_use_view_direction', 
                ]
            cmd = ' '.join(cmd)
            os.system(cmd)
    #         break
    #     break
    # break