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


# retrain: 
# table_with_vase 
# /ssddata/yliugu/trial_model_final/mask_nerf/3dfront_0019_00-bed-nerf 
# /ssddata/yliugu/trial_model_final/mask_nerf/3dfront_0019_00-table-nerf 
# /ssddata/yliugu/trial_model_final/mask_nerf/3dfront_0089_00-large_sofa-nerf 
# /ssddata/yliugu/trial_model_final/mask_nerf/3dfront_0091_00-bed-nerf 
# /ssddata/yliugu/trial_model_final/mask_nerf/donuts-donut_1-nerf  
# /ssddata/yliugu/trial_model_final/mask_nerf/room-chair-nerf  
# /ssddata/yliugu/trial_model_final/mask_nerf/teatime-sheep-nerf


# problem: 
# /ssddata/yliugu/trial_model_final/mask_nerf/cecread-human-nerf 
# /ssddata/yliugu/trial_model_final/mask_nerf/colinepiano-human-nerf 
# /ssddata/yliugu/trial_model_final/mask_nerf/fortress-fortress-nerf  
# /ssddata/yliugu/trial_model_final/mask_nerf/fenceflower-fenceflower-nerf 
# /ssddata/yliugu/trial_model_final/mask_nerf/fern-fern-nerf


    
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
