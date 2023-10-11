import numpy as np
import os
import json
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

scene_list = '/ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/scene_list.json'
workspace = '/ssddata/yliugu/Segment-Anything-NeRF/trial_model_final'
with open(scene_list) as f:
    scene_list = json.load(f)
    
scene_root = '/ssddata/yliugu/data'
scene_list = scene_list['lift'][-1:]

for scene in tqdm(scene_list):
    cur_scene_root = os.path.join(scene_root, scene)

    cmd = [ 'python main.py', 
            cur_scene_root,
            '--workspace',
            os.path.join(workspace, 'rgb_nerf', scene),
            '--enable_cam_center',
            '--downscale 1',
            '--data_type lift ',
            '--iters 20000',
            '--contract',
            '--val_type default',
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
            '--data_type lift',
            '--iters 10000',
            '--contract', 
            '--val_type default',
            '--sam_use_view_direction'
           ]
    cmd = ' '.join(cmd)
    os.system(cmd)
 