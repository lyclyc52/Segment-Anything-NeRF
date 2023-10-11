import numpy as np
import os
import json
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

scene_list = '/ssddata/yliugu/Segment-Anything-NeRF/trial_model_final/scene_list.json'
workspace = '/ssddata/yliugu/Segment-Anything-NeRF/trial_model_final'
with open(scene_list) as f:
    scene_list = json.load(f)
    
scene_root = '/ssddata/yliugu/data'
scene_list = ["horns"]

for scene in tqdm(scene_list):
    cur_scene_root = os.path.join(scene_root, scene)
    
    candidate_paths = [
            os.path.join(cur_scene_root, "colmap_sparse", "0"),
            os.path.join(cur_scene_root, "sparse", "0"),
            os.path.join(cur_scene_root, "colmap"),
        ]
        
    has_run_colmap = False
    for path in candidate_paths:
        if os.path.exists(path):
            has_run_colmap = True
            break

    if not has_run_colmap:
        image_root = os.path.join(scene_root, scene, 'images')
        cmd = ["python",
            "scripts/colmap2nerf.py" ,
            "--images" ,
            image_root,
            '--run_colmap']
        cmd = ' '.join(cmd)
        os.system(cmd)
    
    cmd = [ 'python main.py', 
            cur_scene_root,
            '--workspace',
            os.path.join(workspace, 'rgb_nerf', scene),
            '--enable_cam_center',
            '--downscale 1',
            '--data_type mip ',
            '--iters 15000',
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
            '--data_type mip',
            '--iters 5000',
            '--contract', 
            '--sam_use_view_direction'
           ]
    cmd = ' '.join(cmd)
    os.system(cmd)
 