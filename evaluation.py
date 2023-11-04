import cv2
import numpy as np
import os
import os.path as path
import json



def get_image_name_ours(img_root, object_name, scene_name, data_type, img_id):
    img_name = os.path.join(img_root, f'{scene_name}-{object_name}-nerf', 'results', f'{img_id}_mask.npy')
    inference = np.load(img_name)
                    
    inference = inference.argmax(-1)
    return inference

def get_image_name_sa3d(img_root, object_name, scene_name, data_type, img_id):
    if data_type == 'llff':
        root = 'llff'
    else:
        root = 'nerf_unbounded'
    img_name = os.path.join(img_root, root, f'dvgo_{scene_name}', f'render_test_{object_name}', 'seged_img', f'{img_id}.png')
    # print(img_name)
    inference = cv2.imread(img_name).sum(-1) 
    
    inference = inference < (255*3)
    
    return inference

def get_image_name_isrf(img_root, object_name, scene_name, data_type, img_id):

    img_name = os.path.join(img_root, f'{scene_name}_{object_name}', 'test', f'{img_id}.png')
    # print(img_name)
    inference = cv2.imread(img_name)[..., 0] 
    
    inference = inference  >  0
    
    return inference
get_name_fucntion_dict ={
    'ours': get_image_name_ours,
    'sa3d': get_image_name_sa3d,
    'isrf': get_image_name_isrf,
}

def main(model_root, img_root, model_name='ours'):
    

    mask_data_root = '/ssddata/yliugu/selected_masks'
    meta_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_metadata_v2.json'
    scene_path = '/ssddata/yliugu/Segment-Anything-NeRF/scene_list.json'
    eval_scene_path = '/ssddata/yliugu/Segment-Anything-NeRF/scenes_test_view.json'
    
    
    get_name_fucntion = get_name_fucntion_dict[model_name]
    
    with open(scene_path) as f:
        scene_dict = json.load(f)

    with open(meta_path) as f:
        meta = json.load(f)
    
    with open(eval_scene_path) as f:
        eval_scene_json  = json.load(f)
        
        
    for data_type in list(scene_dict.keys()):
        data_type = 'lift'
        scene_list = scene_dict[data_type]
        total_acc = 0
        total_iou = 0
        
        obj_count = 0

        for scene_name in scene_list:
        
            scene_data_root = path.join(mask_data_root, scene_name)
            
            for object_name in meta[scene_name]:         

                gt_mask_folder = path.join(scene_data_root, object_name)
                
                eval_img_names = eval_scene_json[scene_name][object_name]
                
                # if len(eval_img_names) < 10 and data_type != 'llff':
                #     print(scene_name, object_name)
                    
                    
                cur_iou = 0
                cur_acc = 0
                cur_count = 0
                for eval_img in eval_img_names:
                    inference = get_name_fucntion(img_root,object_name,scene_name,data_type,img_id=eval_img)
                    gt_path = path.join(gt_mask_folder, f'{eval_img}_mask.png')
                    

                    gt_img = cv2.imread(gt_path)[..., 0]

                    if inference.shape[0] != gt_img.shape[0]:
                        # print(scene_name)
       
                        assert abs(inference.shape[0] / gt_img.shape[0] - inference.shape[1] / gt_img.shape[1]) < 0.1
                        
                        gt_img = cv2.resize(gt_img, (inference.shape[1], inference.shape[0]))
                        
                    gt_img = gt_img > 128
                    
                    # cv2.imwrite('gt_mask_{eval_img}.png', gt_img*255)
                    # cv2.imwrite('{eval_img}.png', inference*255)
                    print(object_name)
                    print(eval_acc(inference, gt_img))
                    print(eval_iou(inference, gt_img))
                    # print(inference.shape)
                    # print(gt_img.shape)
                    cur_acc += eval_acc(inference, gt_img)
                    cur_iou += eval_iou(inference, gt_img)
                    # cv2.imwrite(path.join(gt_mask_folder, f'pred_mask_{eval_img}.png'), inference*255)
                    cur_count +=1
                

                # print(f'{scene_name}_{object_name} acc: {(cur_acc / cur_count)}')
                # print(f'{scene_name}_{object_name} iou: {(cur_iou / cur_count)}')
                # print()
                obj_count += 1
                total_acc += (cur_acc / cur_count)
                total_iou += (cur_iou / cur_count)
                
                
        print(f'{data_type}:')
        print(f'acc: ', total_acc / obj_count)
        print(f'miou: ', total_iou / obj_count)
        exit()
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
    
    img_root = '/ssddata/yliugu/trial_model_final/mask_nerf'
    model_name = 'ours'
    # img_root = '/ssddata/yliugu/trial_model_final/mask_nerf'
    
    # img_root = '/ssddata/yliugu/SegmentAnythingin3D/logs'
    # model_name = 'sa3d'
    
    # img_root = '/ssddata/yliugu/isrf_code/masks'
    # model_name = 'isrf'
    
    main(model_root, img_root, model_name)
    
    
