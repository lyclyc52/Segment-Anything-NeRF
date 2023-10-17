import numpy as np
import os
import json
from tqdm import tqdm
import os.path as path
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

scene_dict = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
data_root = '/ssddata/yliugu/data'
workspace_root = '/ssddata/yliugu/trial_model_final'
metadata_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_metadata_v2.json'


with open(scene_dict) as f:
    scene_dict = json.load(f)

with open(metadata_path) as f:
    meta = json.load(f)




for data_type in list(scene_dict.keys()):
    
    # data_type = 'mip'
    
    
    scene_list = scene_dict[data_type]

    for scene_name in scene_list:
        
        
        # scene_name = 'garden'
        
        # print(scene_name)
        
        sam_path = path.join(workspace_root, 'sam_nerf', scene_name)
        ckpt_path = path.join(sam_path, 'checkpoints')
        checkpoint_list = sorted(glob.glob(f'{ckpt_path}/*.pth'))
        ckpt_path = checkpoint_list[-1]
        scene_data_root = path.join(data_root, scene_name)
        
        for object_name in meta[scene_name]:
            
            # object_name = 'table_whole'
            
            iters = 1000
            rgb_similarity_iter=600
            workspace_ending = '-rgb'
            
            # use more iterations if their are too many images

            ending = 'nerf'
            use_rgb_loss = [False]
            
            
            for use_loss in use_rgb_loss:
                if not use_loss:
                    rgb_similarity_iter=iters+1
                    workspace_ending = ''
                    
                mask_folder_name = f'train_{object_name}_{ending}'
                
                valid_json = path.join(scene_data_root, mask_folder_name, 'valid.json')
                valid_count = 0
                with open(valid_json) as f:
                    valid_points = json.load(f)
                    for k in list(valid_points.keys()):
                        if valid_points[k] == 1:
                            valid_count += 1

                
                iters = (valid_count // 5) * 7
                rgb_similarity_iter = int(iters * 0.8)
                    
                    

                object_workspace = path.join(workspace_root, 'mask_nerf', f'{scene_name}-{object_name}-{ending}{workspace_ending}')
                if path.isdir(object_workspace):
                    print(f'skip {object_workspace}')
                    continue
                cmd = ['python main.py', scene_data_root,
                        '--mask_folder_name', mask_folder_name,
                        '--workspace', object_workspace,
                        '--init_ckpt', ckpt_path,
                        '--enable_cam_center',
                        '--with_mask',
                        '--data_type', data_type,
                        '--patch_size 1', 
                        '--num_rays 6000',
                        '--iters', str(iters),
                        '--mask_mlp_type default',
                        '--contract',
                        '--rgb_similarity_loss_weight 10', 
                        '--rgb_similarity_threshold 0.15',
                        '--rgb_similarity_iter', str(rgb_similarity_iter), # Enlarge this to disable rgb loss
                        '--rgb_similarity_num_sample 20',
                        '--local_sample_patch_size 16', 
                        '--num_local_sample 16',
                        '--sum_after_mlp', 
                        '--mixed_sampling',
                        '--error_map']
                cmd = ' '.join(cmd)
                os.system(cmd)
    #         break
    #     break
    # break

        