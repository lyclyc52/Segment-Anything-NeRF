import numpy as np
import os
import json
from tqdm import tqdm
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

scene_dict = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
workspace = '/ssddata/yliugu/trial_model_final'
with open(scene_dict) as f:
    scene_dict = json.load(f)
    
scene_root = '/ssddata/yliugu/data'

    
for scene_type in list(scene_dict.keys()):
    
    
    scene_type = 'mip'
    
    for scene in tqdm(scene_dict[scene_type]):
        
        scene = 'garden'

        cur_scene_root = os.path.join(scene_root, scene)
        cmd = ['python main.py', os.path.join(scene_root, scene),
                '--workspace', os.path.join(workspace, 'sam_nerf', scene),
                '--enable_cam_center', 
                '--downscale 1',
                '--data_type', scene_type, 
                '--test', 
                '--test_split val', 
                '--val_type val_all',
                '--with_sam',
                '--num_rays 16384',
                '--contract', 
                '--sam_use_view_direction', 
                '--return_extra'
            ]

        cmd = ' '.join(cmd)
        os.system(cmd)
        break
    break
