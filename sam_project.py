import torch 
import numpy as np 
import json
import os
import cv2
import imageio
import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, sam_model_registry_baseline
import torch.nn.functional as F
import argparse
import json



os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def downsample_sam_feature(feature):
    feature_max_dim = np.max(feature.shape[1:])
    padded_feature = np.pad(feature, ((0, 0), (0, feature_max_dim - feature.shape[1]), 
                                      (0, feature_max_dim - feature.shape[2])), mode='constant')
    
    target_res = 64
    factor = feature_max_dim // target_res
    offset = (feature_max_dim - factor * target_res) 
    index = torch.linspace(0, end=feature_max_dim-1, steps=target_res).to(torch.int)

    if factor > 1:
        # downsampled_feature = padded_feature[:, offset::factor, offset::factor]
        # downsampled_feature = downsampled_feature[:, :64, :64]
        downsampled_feature = padded_feature[:, index]
        downsampled_feature = downsampled_feature[:, :, index]
    else:
        downsampled_feature =padded_feature[:, :64, : 64]

    return downsampled_feature


def main(args, pts_3D, input_label):
    sam_checkpoint = args.sam_checkpoint
    device = args.device
    model_type = args.model_type
    threshold = args.threshold

    pose_file = args.pose_file
    frame_root = args.frame_root
    camera_poses = args.camera_poses
    output_root = args.output_root

    


    os.makedirs(output_root, exist_ok=True)
    with open(pose_file) as f:
        poses = json.load(f)
    
    frame_names = []
    for k in poses.keys():
        frame_names.append(k)
    # print(frame_names)
    # exit()

    H = W = 512
    fovy = 60
    focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
    intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)


    if args.sam_type == 'sam':
        sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
    elif args.sam_type == 'hq_sam':
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    
    sam.to(device=device)
    predictor = SamPredictor(sam)



    rgb_name = os.path.join(frame_root, f'ngp_ep{args.epoch}_{frame_names[0]}_rgb.png')
    
    image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    


    # pts_3D = torch.tensor([[-0.0873, -0.2738, -0.3423],
    #                         [ 0.0501, -0.3104, -0.3342],
    #                         [ 0.0146, -0.2055, -0.3980],
    #                         [ 0.0706, -0.3154, -0.1454],
    #                         [ 0.0385, -0.4497, -0.3302],
    #                         [-0.0648, -0.4514, -0.2964],
    #                         [-0.1149, -0.5198, -0.3918],
    #                         [-0.0710, -0.3939, -0.6091]])
    
    
    
    
    valid_count = {}
    
    for i in tqdm.tqdm(range(len(frame_names))):


        rgb_name = os.path.join(frame_root, f'ngp_ep{args.epoch}_{frame_names[i]}_rgb.png')
        depth_name = os.path.join(frame_root, f'ngp_ep{args.epoch}_{frame_names[i]}_depth.npy')
        feature_name = os.path.join(frame_root, f'ngp_ep{args.epoch}_{frame_names[i]}.npy')
        
        image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
        r = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        p,n = poses[frame_names[i]], intrinsics
        
        p = np.array(p).astype(np.float32)
        # p[:,1] = -p[:, 1]
        # p[:,2] = -p[:, 2]
        output_file = os.path.join(output_root, f'{frame_names[i]}_masks.npy')
      

        d = np.load(depth_name)[..., None]

        pts_ids = np.arange(pts_3D.shape[0])
   
        pts_2D, pts_depth = project_to_2d(pts_3D, p, n, H = r.shape[0], W = r.shape[1])
        
        valid =  np.logical_and.reduce((pts_2D[..., 1] < d.shape[0], pts_2D[..., 0] < d.shape[1],
            pts_2D[..., 1] >= 0, pts_2D[..., 0] >=0))
        
        if valid.sum() < 1:
            # print(f"{frame_names[i]} fails")
            mask = np.zeros(d.shape)
            np.save(output_file, mask)
            
            output_file = os.path.join(output_root, f'{frame_names[i]}_masks.png')
            plt.figure(figsize=(10,10))
            plt.imshow(r)
            plt.axis('on')
            plt.savefig(output_file)
            plt.close()
            valid_count[frame_names[i]] = 0
            continue
        pts_2D = pts_2D[valid]
        pts_depth = pts_depth[valid]
        pts_label = input_label[valid]
        pts_ids = pts_ids[valid]


        im_depth = torch.from_numpy(d[pts_2D[...,1], pts_2D[..., 0]])

        if im_depth.shape[0] >= 2:
            im_depth = im_depth.transpose(1,0)
            im_depth = im_depth[0]
        # print((torch.abs(im_depth- pts_depth)  ))
        valid = np.logical_and.reduce((pts_depth > 0., torch.abs(im_depth- pts_depth) < threshold))
        
        if valid.sum() < 1:
            # print(f"{frame_names[i]} fails")
            mask = np.zeros(d.shape)
            np.save(output_file, mask)
            
            output_file = os.path.join(output_root, f'{frame_names[i]}_masks.png')
            plt.figure(figsize=(10,10))
            plt.imshow(r)
            plt.axis('on')
            plt.savefig(output_file)
            plt.close()
            valid_count[frame_names[i]] = 0
            continue
        
        pts_2D = pts_2D[valid]
        pts_label = pts_label[valid]
        pts_ids = pts_ids[valid]
        
        
        
        if args.decode:
            if args.use_nerf_feature:

                f = np.load(feature_name)[0]
                f = downsample_sam_feature(f)
                predictor.features = torch.from_numpy(f).to(predictor.device)[None, ...]
            else:
                predictor.set_image(r)
            
            masks, scores, logits = predictor.predict(
                point_coords=pts_2D.numpy(),
                point_labels=pts_label,
                multimask_output=True,
                hq_token_only=True
            )
            
            max_score = 0
            index = 0
            for j, (mask, score) in enumerate(zip(masks, scores)):
                if score > max_score:
                    max_score = score
                    index = j
            
            # print(masks[index][..., None].shape)
            np.save(output_file, masks[index][..., None])
            output_file = os.path.join(output_root, f'{frame_names[i]}_masks.png')
            plt.figure(figsize=(10,10))
            plt.imshow(r)
            show_mask(masks[index], plt.gca())
            show_points(pts_2D, pts_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {scores[index]:.3f}", fontsize=18)
            plt.axis('on')
            plt.savefig(output_file)
            plt.close()
            # for pp in range(pts_ids.shape[0]):
            #     print(pts_ids[pp])
            #     show_points(pts_2D[pp:pp+1], pts_label[pp:pp+1], plt.gca())
            #     plt.title(f"Mask {i+1}, Score: {scores[index]:.3f}", fontsize=18)
            #     plt.axis('on')
            #     plt.savefig(output_file.replace('.png', f'{pp}.png' ))
                
            # exit()
        
        crucial = False
        if args.crucial_point_index is None:
            crucial = True
        else:
            for j in args.crucial_point_index:
                if j in pts_ids:
                    crucial = True
                    break
        if valid.sum() >= args.valid_threohould and crucial:
            valid_count[frame_names[i]] = 1
        else:
            valid_count[frame_names[i]] = 0

        
        
        
    valid_file = os.path.join(output_root, 'valid.json')
    with open(valid_file, 'w') as f:
        f.write(json.dumps(valid_count, indent=4))




# def get_3d_pts():

#     sam_checkpoint = "/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_vit_h_4b8939.pth"
#     device = "cuda:3"
#     model_type = "vit_h"
#     threshold = 0.05
#     # frame_root = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/results/frames'
#     pose_file = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/pose_dir.json'
#     frame_root = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/validation'
#     camera_poses = os.path.join(frame_root, 'camera.npz')

#     output_root = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/masks'

#     input_point = np.array([[400, 500]])
#     input_label = np.array([1])
    
    
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     predictor = SamPredictor(sam)


#     os.makedirs(output_root, exist_ok=True)
#     with open(pose_file) as f:
#         poses = json.load(f)
    
#     frame_names = []
#     for k in poses.keys():
#         frame_names.append(k)
    

#     H = W = 512
#     fovy = 60
#     focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
#     intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)


#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     predictor = SamPredictor(sam)
#     pts_3D = None

#     rgb_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[0]}_rgb.png')
#     image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     predictor.set_image(image)
    
    



#     # print(extrinsics.shape, intrinsics.shape, images.shape, depths.shape, features.shape)
#     for i in tqdm.tqdm(range(len(frame_names))):
#         i=100
#         frame_names[i] = 'frame_00019'
        
#         rgb_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_rgb.png')
#         depth_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_depth.npy')
#         feature_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_sam.npy')
#         p,n = poses[frame_names[i]], intrinsics
#         p = np.array(p).astype(np.float32)
#         p[:,1] = -p[:, 1]
#         p[:,2] = -p[:, 2]
      
#         image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
#         r = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         d = np.load(depth_name)[..., None]
#         f = np.load(feature_name)[0]


        # pts_3D = project_to_3d(input_point, p, n, d)
#         pts_2D, pts_depth = project_to_2d(pts_3D, p, n)
        
#         # valid = pts_2D[..., 1] < r.shape[0] and pts_2D[..., 0] < r.shape[1] and \
#         #     pts_2D[..., 1] >= 0 and pts_2D[..., 0] >=0
#         valid =  np.logical_and.reduce((pts_2D[..., 1] < r.shape[0], pts_2D[..., 0] < r.shape[1],
#             pts_2D[..., 1] >= 0, pts_2D[..., 0] >=0))

#         if valid.sum() < 1:
#             print(f"{i} fails")
#             mask = np.zeros(d.shape)
#             np.save(output_file, mask)
#             continue
#         pts_2D = pts_2D[valid]
#         pts_depth = pts_depth[valid]
#         pts_label = input_label[valid]


#         im_depth = torch.from_numpy(d[pts_2D[...,1], pts_2D[..., 0]])

#         if im_depth.shape[0] >= 2:
#             im_depth = im_depth.transpose(1,0)
#             im_depth = im_depth[0]
#         # print((torch.abs(im_depth- pts_depth)  ))
#         valid = np.logical_and.reduce((pts_depth > 0., torch.abs(im_depth- pts_depth) < threshold))
        
#         if valid.sum() < 1:
#             print(f"{i} fails")
#             mask = np.zeros(d.shape)
#             np.save(output_file, mask)
#             continue
        
#         pts_2D = pts_2D[valid]
#         pts_label = pts_label[valid]
        
        
#         f = downsample_sam_feature(f)
#         predictor.features = torch.from_numpy(f).to(predictor.device)[None, ...]
        
#         masks, scores, logits = predictor.predict(
#             point_coords=pts_2D.numpy(),
#             point_labels=pts_label,
#             multimask_output=True,
#         )

#         for j, (mask, score) in enumerate(zip(masks, scores)):
#             output_file = os.path.join(output_root, f'test_{frame_names[i]}_masks_{j}.png')
#             plt.figure(figsize=(10,10))
#             plt.imshow(r)
#             show_mask(mask, plt.gca())
#             show_points(pts_2D, pts_label, plt.gca())
#             plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#             plt.axis('on')
#             plt.savefig(output_file)
#             plt.close()
#         break







def project_to_3d(pts, pose, intrinsics, depth):
    '''
    Args:
        pts: Nx2
        pose: 4x4
        intrinsics: fx, fy, cx, cy
        depth: HxW
    '''
    pose[:,1] = -pose[:, 1]
    pose[:,2] = -pose[:, 2]
    
    pts = torch.from_numpy(pts)
    pose = torch.tensor(pose)
    fx, fy, cx, cy = intrinsics
    zs = torch.ones_like(pts[..., 0])
    xs = (pts[..., 0] - cx) / fx * zs
    ys = (pts[..., 1]  - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    pts_z = depth[pts[..., 1], pts[..., 0]] 
    directions = directions * pts_z
    
    rays_d = directions @ pose[:3, :3].transpose(1,0) # (N, 3)
    rays_o = pose[:3, 3] # [3]
    rays_o = rays_o[None, :]
    return rays_o + rays_d

def project_to_2d(pts, pose, intrinsics, H, W):

    fx, fy, cx, cy = intrinsics
    pose = torch.tensor(pose)
    w2c = torch.inverse(pose)

    
    pts = torch.concat([pts, torch.ones([pts.size(0), 1])], -1)
    camera_pts = pts @ w2c.T
    camera_pts = camera_pts[..., :-1]
    

    camera_x = W - (camera_pts[..., 0] / camera_pts[..., 2] * fx + cx)
    camera_y = camera_pts[..., 1] / camera_pts[..., 2] * fy + cy
    
    # pts_depth = torch.norm(camera_pts, dim=-1)
    # pts_depth = pts_depth 

    
    pts_depth = -camera_pts[..., 2]
    
    return torch.stack([camera_x, camera_y], dim = -1).to(torch.int), pts_depth

def create_video():
    input_path = '/disk1/yliugu/Grounded-Segment-Anything/outputs/3dfront_0089/video'
    save_path = '/disk1/yliugu/Grounded-Segment-Anything/outputs/3dfront_0089/video'
    all_preds = []
    index = 0
    for i in range(11):
        image = cv2.imread(os.path.join(input_path, f'{i:04d}_masks_{index}.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_preds.append(image)
    
    imageio.mimwrite(os.path.join(save_path, f'rgb_{index}.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--sam_checkpoint', type=str, default='/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_hq_vit_h.pth')
    # parser.add_argument('--sam_checkpoint', type=str, default='/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', type=str,
                        default='sam', choices=['sam', 'hq_sam'])
    

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_type', type=str, default='vit_h')
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--epoch', type=str, default='0040')
    parser.add_argument('--scene_name', type=str, default='shoe_rack')
    parser.add_argument('--scene_object', type=str, default='blue_shoes')
    parser.add_argument('--valid_threohould', type=int, default=0)

    parser.add_argument('--files_root', type=str, default='/ssddata/yliugu/Segment-Anything-NeRF/trial_model')
    parser.add_argument('--purpose',  type=str,
                        default='eval', choices=['train', 'eval'])
    
    parser.add_argument('--output_root', type=str, default='/ssddata/yliugu/data')
    parser.add_argument('--output_folder_ending', type=str, default='')
    parser.add_argument('--use_nerf_feature', action='store_true', help='use nerf-rendered feature to obtain the mask')
    parser.add_argument('--decode', action='store_true', help='generate mask if true')
    parser.add_argument('--crucial_point_index', type=int, nargs='*', default=None, help="num steps sampled per ray for each proposal level")
    


    args = parser.parse_args()
    
    args.files_root = os.path.join(args.files_root, f'trial_{args.scene_name}_sam')
    if args.purpose == 'eval':
        args.files_root = os.path.join(args.files_root, 'results')
    args.pose_file = os.path.join(args.files_root, 'pose_dir.json')
    args.frame_root = args.files_root
    args.camera_poses = os.path.join(args.frame_root, 'camera.npz')
    
    
    
    scene_object_file = args.sam_type
    if args.use_nerf_feature:
        scene_object_file = 'nerf'
    args.output_root = os.path.join(args.output_root, args.scene_name, f'{args.purpose}_{args.scene_object}_{scene_object_file}')
    
    if args.sam_type == 'sam':
        args.sam_checkpoint = '/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_vit_h_4b8939.pth'
    elif args.sam_type == 'hq_sam':
        args.sam_checkpoint = '/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_hq_vit_h.pth'
        # parser.add_argument('--sam_checkpoint', type=str, default='/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_hq_vit_h.pth')
    # parser.add_argument('--sam_checkpoint', type=str, default='/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_vit_h_4b8939.pth')

    
    # load reference points:
    with open('scene_dict.json') as scen_dict_file:
        scene_dict = json.load(scen_dict_file)
        pts_3D = scene_dict[args.scene_name][args.scene_object]['points']
        pts_3D = torch.tensor(pts_3D)
        input_label = np.ones(pts_3D.shape[0])
        for i in scene_dict[args.scene_name][args.scene_object]['negative_labels']:
            input_label[i] = 0
    
    
    # # test case 0
    # pts_3D = torch.tensor([[-0.0956, -0.4185, -0.3230],
    #                         [-0.0135, -0.3071, -0.1701],
    #                         [ 0.0014, -0.2761, -0.3728],
    #                         [-0.1080, -0.2990, -0.3226],
    #                         [-0.1104, -0.5073, -0.3695],
    #                         [ 0.0226, -0.3977, -0.2522],
    #                         [-0.0048, -0.5092, -0.3536],
    #                         [ 0.0386, -0.4081, -0.4083],
    #                         [ 0.0169, -0.4760, -0.4811],
    #                         [-0.0047, -0.1596, -0.4147]])

    # input_label = np.ones(pts_3D.shape[0])
    # input_label[-1] = 0 
    
    
    # test case 2: sheep
    # pts_3D = torch.tensor([[ 0.0349, -0.2395, -0.1510],
    #                         [ 0.0265, -0.2429, -0.3851],
    #                         [-0.1043, -0.2565, -0.3524],
    #                         [ 0.0607, -0.3543, -0.2073],
    #                         [-0.0913, -0.4343, -0.3219],
    #                         [-0.1727, -0.4455, -0.5075],
    #                         [-0.0554, -0.4103, -0.5615],
    #                         [ 0.0615, -0.4076, -0.3355],
    #                         [-0.0432, -0.3506, -0.4291]])
    # input_label = np.ones(pts_3D.shape[0])
    
    
    # # test case 4: garden
    # pts_3D = torch.tensor([[ 0.0922, -0.1093, -0.4020],
    #     [ 0.0973,  0.2045, -0.3965],
    #     [ 0.1441,  0.1804, -0.5945],
    #     [ 0.0557, -0.1154, -0.6349],
    #     [-0.2492, -0.0210, -0.6243],
    #     [-0.1622,  0.2775, -0.6516]])
    # input_label = np.ones(pts_3D.shape[0])
    
    
    # # test case 4: garden
    # pts_3D = torch.tensor([[ 0.1772,  0.0180, -0.4012],
    #                     [-0.1294,  0.2618, -0.3944],
    #                     [-0.1300,  0.0416, -0.4265],
    #                     [-0.1215,  0.0604, -0.4093],
    #                     [-0.0517,  0.0580, -0.6673],
    #                     [ 0.0532, -0.1099, -0.6656],
    #                     [ 0.1468,  0.1817, -0.6530],
    #                     # [-0.1656,  0.2856, -0.6465],
    #                     # [-0.2695, -0.0262, -0.6177],
    #                     # negative points
    #                     [-0.0422,  0.0305, -0.3319],
    #                     [-0.0534,  0.1006, -0.3437]])
    # input_label = np.ones(pts_3D.shape[0])
    
    # input_label[-2:] = 0
    
    
    # test case 5: garden
    # pts_3D = torch.tensor([[ 0.1554, -0.0071, -0.4008],
    #                     [-0.2675,  0.1471, -0.3918],
    #                     [-0.0962,  0.0480, -0.4486],
    #                     [-0.0503,  0.0809, -0.6661],
    #                     [-0.0384,  0.0431, -0.2802],
    #                     [-0.0732,  0.0417, -0.2275]])
    # input_label = np.ones(pts_3D.shape[0])
    
    # input_label[-2:] = 0

    
    
    

    
    main(args, pts_3D, input_label)
    # get_3d_pts()
    # create_video()