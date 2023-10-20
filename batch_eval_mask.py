import numpy as np
import os
import json
from tqdm import tqdm
import shutil
import os.path as path

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

scene_dict = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
workspace = '/ssddata/yliugu/trial_model_final'
with open(scene_dict) as f:
    scene_dict = json.load(f)
    
scene_root = '/ssddata/yliugu/data'



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

scene_dict = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
data_root = '/ssddata/yliugu/data'
workspace_root = '/ssddata/yliugu/trial_model_final'
metadata_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_metadata_v2.json'


with open(scene_dict) as f:
    scene_dict = json.load(f)

with open(metadata_path) as f:
    meta = json.load(f)


scene_dict = {}
scene_dict['lift'] = ['ctr_lift_0', 'ctr_lift_1', 'ctr_lift_2']

for data_type in list(scene_dict.keys()):
    
    # data_type = 'llff'
    
    
    scene_list = scene_dict[data_type]

    for scene_name in scene_list:
        
        
        # scene_name = 'fenceflower'
        
        # print(scene_name)
        
        # sam_path = path.join(workspace_root, 'sam_nerf', scene_name)
        # ckpt_path = path.join(sam_path, 'checkpoints')
        # checkpoint_list = sorted(glob.glob(f'{ckpt_path}/*.pth'))
        # ckpt_path = checkpoint_list[-1]
        scene_data_root = path.join(data_root, scene_name)
        
        for object_name in meta[scene_name]:
        
            ending = 'nerf'
            cur_scene_root = os.path.join(scene_root, scene_name)
            object_workspace = path.join(workspace_root, 'mask_nerf', f'{scene_name}-{object_name}-{ending}')
            mask_folder_name = f'train_{object_name}_{ending}'
            
            with open(path.join(cur_scene_root, mask_folder_name, 'data_split.json')) as f:
                eval_json = json.load(f)
            # if 'use_default_intrinstic' not in eval_json or eval_json['use_default_intrinstic'] != 1:
            #     continue
            
            cmd = ['python main.py', cur_scene_root,
                    '--workspace', object_workspace,
                    '--data_type', data_type, 
                    '--mask_folder_name', mask_folder_name,
                    '--enable_cam_center', 
                    '--downscale 1',
                    '--test',
                    '--H 512',
                    '--test_split val', 
                    '--val_type val_split',
                    '--with_mask',
                    '--contract', 
                    '--n_inst 2',
                    '--sam_use_view_direction', 
                    '--render_mask_instance_id -1',
                    '--return_extra'
                ]



            cmd = ' '.join(cmd)
            os.system(cmd)
    #         break
    #     break
    # break
