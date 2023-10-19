import cv2
import numpy as np
import os
import os.path as path
import json

def main(model_root):
    
    data_root = '/ssddata/yliugu/data'
    meta_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_metadata_v2.json'
    scene_path = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
    workspace_root = '/ssddata/yliugu/trial_model_final'
    
    
    with open(scene_path) as f:
        scene_dict = json.load(f)

    with open(meta_path) as f:
        meta = json.load(f)
        
        
    for data_type in list(scene_dict.keys()):
        scene_list = scene_dict[data_type]
        total_acc = 0
        total_iou = 0
        
        count = 0

        for scene_name in scene_list:
            # scene_name='fenceflower'
            scene_data_root = path.join(data_root, scene_name)
            
            for object_name in meta[scene_name]:
                ending = 'nerf'
                
                object_workspace = path.join(workspace_root, 'mask_nerf', f'{scene_name}-{object_name}-{ending}')
                mask_folder_name = f'train_{object_name}_{ending}'
                gt_mask_folder = path.join(scene_data_root, f'eval_{object_name}')
                
                gt_mask_json = path.join(scene_data_root, mask_folder_name, 'data_split.json')
                
                
                with open(gt_mask_json) as f:
                    eval_img_names = json.load(f)
                    eval_img_names = eval_img_names['test']
                
                if len(eval_img_names) < 10 and data_type != 'llff':
                    print(scene_name, object_name)
                    
                    
                    
                for eval_img in eval_img_names:
                    inference_path = path.join(object_workspace, 'results', f'{eval_img}_mask.npy')
                    gt_path = path.join(gt_mask_folder, f'refined_mask_{eval_img}.png')
                    inference = np.load(inference_path)
                    
                    inference = inference.argmax(-1)
                    
                    gt_img = cv2.imread(gt_path)[..., 0]
                    if inference.shape[0] != gt_img.shape[0]:
                        print(scene_name)
                        gt_img = cv2.resize(gt_img, (inference.shape[1], inference.shape[0]))
                        
                    gt_img = gt_img > 128
                    
                    total_acc += eval_acc(inference, gt_img)
                    total_iou += eval_iou(inference, gt_img)
                    cv2.imwrite(path.join(gt_mask_folder, f'pred_mask_{eval_img}.png'), inference*255)
                    count +=1
                    
                    
                
        print(f'{data_type}:')
        print(f'acc: ', total_acc / count)
        print(f'miou: ', total_iou / count)
    return

def eval_iou(inference, gt):
    intersection = (inference * gt).sum()
    union = ((inference + gt) > 0).sum()
    if union == 0:
        if intersection == 0:
            return 1
        else:
            return 0
    return intersection / union

def eval_acc(inference, gt):
    
    inference_flatten = inference.reshape(-1)
    gt_flatten = gt.reshape(-1)
    
    false_pred = np.logical_xor(inference_flatten, gt_flatten).sum()
    total = inference_flatten.shape[0]
    
    
    return 1. - false_pred/ total

if __name__ == '__main__':
    model_root = '/ssddata/yliugu/trial_model_final/mask_nerf'
    main(model_root)