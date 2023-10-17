import numpy as np
import os
import json
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

scene_dict = '/ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/scene_list.json'
workspace = '/ssddata/yliugu/Segment-Anything-NeRF/trial_model_final'
with open(scene_dict) as f:
    scene_dict = json.load(f)
    
scene_root = '/ssddata/yliugu/data'
scene_list = scene_dict['3dfront']

for scene in tqdm(scene_list):
    cur_scene_root = os.path.join(scene_root, scene)
    
    
    file_path = os.path.join(workspace, 'sam_nerf', scene)
    if os.path.isfile(file_path):
        continue
        
    cmd = [ 'python main.py', 
            cur_scene_root,
            '--workspace',
            os.path.join(workspace, 'rgb_nerf', scene),
            '--enable_cam_center',
            '--downscale 1',
            '--data_type 3dfront ',
            '--iters 5000',
            '--contract',
            '--random_image_batch']
    cmd = ' '.join(cmd)
    os.system(cmd)


    cmd = [ 'python main.py ',
            cur_scene_root,
            '--workspace',
            os.path.join(workspace, 'sam_nerf', scene),
            '--with_sam', 
            '--init_ckpt',
            os.path.join(workspace, 'rgb_nerf', scene, 'checkpoints' , 'ngp.pth'), 
            '--data_type 3dfront',
            '--iters 5000',
            '--contract', 
            '--sam_use_view_direction'
           ]
    cmd = ' '.join(cmd)
    os.system(cmd)
