import torch 
import numpy as np 
import json
import os
import cv2
import imageio
import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F

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


def main():
    sam_checkpoint = "/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_vit_h_4b8939.pth"
    device = "cuda:3"
    model_type = "vit_h"
    threshold = 0.1
    # frame_root = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/results/frames'
    pose_file = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/pose_dir.json'
    frame_root = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/validation'
    camera_poses = os.path.join(frame_root, 'camera.npz')

    output_root = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/masks'

    input_point = np.array([[100, 330]])
    input_label = np.array([1])
    
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    os.makedirs(output_root, exist_ok=True)
    with open(pose_file) as f:
        poses = json.load(f)
    
    frame_names = []
    for k in poses.keys():
        frame_names.append(k)
    

    H = W = 512
    fovy = 60
    focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
    intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    pts_3D = None

    rgb_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[0]}_rgb.png')
    image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    
    pts_3D = torch.tensor([[0.1217, -0.2355, -0.2479],
                           [0.1513, -0.3258, -0.2918],
                           [0.0298, -0.2842, -0.3166],
                           [-0.1392, -0.3707, -0.1720],
                           [-0.0608, -0.4495, -0.2166],
                           [ 0.0748, -0.0510, -0.6848]])

    input_label = np.ones(pts_3D.shape[0])
    # print(extrinsics.shape, intrinsics.shape, images.shape, depths.shape, features.shape)
    for i in tqdm.tqdm(range(len(frame_names))):

        rgb_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_rgb.png')
        depth_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_depth.npy')
        feature_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_sam.npy')
        p,n = poses[frame_names[i]], intrinsics
        p = np.array(p).astype(np.float32)
        p[:,1] = -p[:, 1]
        p[:,2] = -p[:, 2]
        output_file = os.path.join(output_root, f'{frame_names[i]}_masks.npy')
      
        image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
        r = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        d = np.load(depth_name)[..., None]
        f = np.load(feature_name)[0]

        pts_2D, pts_depth = project_to_2d(pts_3D, p, n)
        
        # valid = pts_2D[..., 1] < r.shape[0] and pts_2D[..., 0] < r.shape[1] and \
        #     pts_2D[..., 1] >= 0 and pts_2D[..., 0] >=0
        valid =  np.logical_and.reduce((pts_2D[..., 1] < r.shape[0], pts_2D[..., 0] < r.shape[1],
            pts_2D[..., 1] >= 0, pts_2D[..., 0] >=0))

        if valid.sum() < 1:
            print(f"{frame_names[i]} fails")
            mask = np.zeros(d.shape)
            np.save(output_file, mask)
            
            output_file = os.path.join(output_root, f'{frame_names[i]}_masks.png')
            plt.figure(figsize=(10,10))
            plt.imshow(r)
            plt.axis('on')
            plt.savefig(output_file)
            plt.close()
            continue
        pts_2D = pts_2D[valid]
        pts_depth = pts_depth[valid]
        pts_label = input_label[valid]


        im_depth = torch.from_numpy(d[pts_2D[...,1], pts_2D[..., 0]])

        if im_depth.shape[0] >= 2:
            im_depth = im_depth.transpose(1,0)
            im_depth = im_depth[0]
        # print((torch.abs(im_depth- pts_depth)  ))
        valid = np.logical_and.reduce((pts_depth > 0., torch.abs(im_depth- pts_depth) < threshold))
        
        if valid.sum() < 1:
            print(f"{frame_names[i]} fails")
            mask = np.zeros(d.shape)
            np.save(output_file, mask)
            
            output_file = os.path.join(output_root, f'{frame_names[i]}_masks.png')
            plt.figure(figsize=(10,10))
            plt.imshow(r)
            plt.axis('on')
            plt.savefig(output_file)
            plt.close()
            continue
        
        pts_2D = pts_2D[valid]
        pts_label = pts_label[valid]
        
        
        f = downsample_sam_feature(f)
        predictor.features = torch.from_numpy(f).to(predictor.device)[None, ...]
        
        masks, scores, logits = predictor.predict(
            point_coords=pts_2D.numpy(),
            point_labels=pts_label,
            multimask_output=True,
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





def get_3d_pts():

    sam_checkpoint = "/ssddata/yliugu/Segment-Anything-NeRF/pretrained/sam_vit_h_4b8939.pth"
    device = "cuda:3"
    model_type = "vit_h"
    threshold = 0.05
    # frame_root = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/results/frames'
    pose_file = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/pose_dir.json'
    frame_root = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/validation'
    camera_poses = os.path.join(frame_root, 'camera.npz')

    output_root = '/ssddata/yliugu/Segment-Anything-NeRF/trial2_teatime/masks'

    input_point = np.array([[400, 500]])
    input_label = np.array([1])
    
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    os.makedirs(output_root, exist_ok=True)
    with open(pose_file) as f:
        poses = json.load(f)
    
    frame_names = []
    for k in poses.keys():
        frame_names.append(k)
    

    H = W = 512
    fovy = 60
    focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
    intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    pts_3D = None

    rgb_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[0]}_rgb.png')
    image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    



    # print(extrinsics.shape, intrinsics.shape, images.shape, depths.shape, features.shape)
    for i in tqdm.tqdm(range(len(frame_names))):
        i=100
        frame_names[i] = 'frame_00019'
        
        rgb_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_rgb.png')
        depth_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_depth.npy')
        feature_name = os.path.join(frame_root, f'ngp_ep0017_{frame_names[i]}_sam.npy')
        p,n = poses[frame_names[i]], intrinsics
        p = np.array(p).astype(np.float32)
        p[:,1] = -p[:, 1]
        p[:,2] = -p[:, 2]
      
        image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
        r = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        d = np.load(depth_name)[..., None]
        f = np.load(feature_name)[0]


        pts_3D = project_to_3d(input_point, p, n, d)
        pts_2D, pts_depth = project_to_2d(pts_3D, p, n)
        
        # valid = pts_2D[..., 1] < r.shape[0] and pts_2D[..., 0] < r.shape[1] and \
        #     pts_2D[..., 1] >= 0 and pts_2D[..., 0] >=0
        valid =  np.logical_and.reduce((pts_2D[..., 1] < r.shape[0], pts_2D[..., 0] < r.shape[1],
            pts_2D[..., 1] >= 0, pts_2D[..., 0] >=0))

        if valid.sum() < 1:
            print(f"{i} fails")
            mask = np.zeros(d.shape)
            np.save(output_file, mask)
            continue
        pts_2D = pts_2D[valid]
        pts_depth = pts_depth[valid]
        pts_label = input_label[valid]


        im_depth = torch.from_numpy(d[pts_2D[...,1], pts_2D[..., 0]])

        if im_depth.shape[0] >= 2:
            im_depth = im_depth.transpose(1,0)
            im_depth = im_depth[0]
        # print((torch.abs(im_depth- pts_depth)  ))
        valid = np.logical_and.reduce((pts_depth > 0., torch.abs(im_depth- pts_depth) < threshold))
        
        if valid.sum() < 1:
            print(f"{i} fails")
            mask = np.zeros(d.shape)
            np.save(output_file, mask)
            continue
        
        pts_2D = pts_2D[valid]
        pts_label = pts_label[valid]
        
        
        f = downsample_sam_feature(f)
        predictor.features = torch.from_numpy(f).to(predictor.device)[None, ...]
        
        masks, scores, logits = predictor.predict(
            point_coords=pts_2D.numpy(),
            point_labels=pts_label,
            multimask_output=True,
        )

        for j, (mask, score) in enumerate(zip(masks, scores)):
            output_file = os.path.join(output_root, f'test_{frame_names[i]}_masks_{j}.png')
            plt.figure(figsize=(10,10))
            plt.imshow(r)
            show_mask(mask, plt.gca())
            show_points(pts_2D, pts_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('on')
            plt.savefig(output_file)
            plt.close()
        print(pts_3D)
        break



# def get_3d_pts():
#     # intrinsics = np.array([400., 400., 320., 240.])
#     sam_checkpoint = "/ssddata/yliugu/lerf/dependencies/sam-hq/pretrained_checkpoint/sam_vit_l_0b3195.pth"
#     device = "cuda:0"
#     model_type = "vit_l"
#     threshold = 0.05
#     # frame_root = '/disk1/yliugu/torch-ngp/workspace/3dfront_0089_t0_results/results/frames'
#     pose_file = '/ssddata/yliugu/data/Datasets/teatime/transforms.json'
#     frame_root = 'render_feature'
#     camera_poses = os.path.join(frame_root, 'camera.npz')

#     output_root = 'masks'

#     input_point = np.array([[800, 660]])
#     input_label = np.array([1])
    
    
#     sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     predictor = SamPredictor(sam)


#     os.makedirs(output_root, exist_ok=True)
#     with open(pose_file) as f:
#         poses = json.load(f)['frames']
    
#     with np.load(camera_poses) as p:

        
#         intrinsics = np.concatenate([p['fx'], p['fy'], p['cx'], p['cy']], -1).astype(np.float32)
#         print(intrinsics.shape)
#         # c2w = np.array(p['transform_matrix']).astype(np.float32)
        
#         extrinsics = p['camera_to_worlds']
#         # print(extrinsics[0])
#         extrinsics[:, :,1] = -extrinsics[:, :, 1]
#         # print(extrinsics[0])
#         extrinsics[:, :,2] = -extrinsics[:, :, 2]
        
#         row = np.array([[[0,0,0,1]]]).repeat(extrinsics.shape[0], 0)
#         extrinsics = np.concatenate([extrinsics, row], 1).astype(np.float32)
       
        
#     # exit()
    
#     images = []
#     depths = []
#     features = []
    
#     idx = 175
#     extrinsics = extrinsics[idx:idx+1]
#     intrinsics = intrinsics[idx:idx+1]
#     # for i in tqdm.tqdm(range(len(extrinsics))):
#     for i in tqdm.tqdm(range(idx,idx+1)):
    
#         rgb_name = os.path.join(frame_root, f'{i:04}_rgb.png')
#         depth_name = os.path.join(frame_root, f'{i:04}_depth.npy')
#         feature_name = os.path.join(frame_root, f'{i:04}_feature.npy')
#         image = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         images.append(image)
        
#         depths.append(np.load(depth_name))
#         # with np.load(depth_name) as data:
#         #     size = data['size']
#         #     depth = data['depth']
#         #     depth = depth.reshape(size.tolist()).astype(np.float32)
#         #     depths.append(depth)
        
#         features.append(np.load(feature_name))
#         # with np.load(feature_name) as data:
#         #     res = data['res']
#         #     feature = data['embedding']
#         #     feature = feature.reshape(res.tolist()).astype(np.float32)
#         #     features.append(feature)
    
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     predictor = SamPredictor(sam)
#     pts_3D = None
#     predictor.set_image(images[0])
    
#     # pts_3D = project_to_3d(input_point, p, n, d)
    
#     pts_3D = torch.tensor([[-0.1268, -0.3349, -0.3373]])
#     for i, (p,n,r,d,f) in tqdm.tqdm(enumerate(zip(extrinsics, intrinsics, images, depths, features))):
#         if i == 0:
#             pts_3D = project_to_3d(input_point, p, n, d)
            

#         pts_2D, pts_depth = project_to_2d(pts_3D, p, n)
        
#         valid = pts_2D[..., 1] < r.shape[0] and pts_2D[..., 0] < r.shape[1] and \
#             pts_2D[..., 1] >= 0 and pts_2D[..., 0] >=0
            
#         if valid.sum() < 1:
#             continue
#         pts_2D = pts_2D[valid]
#         im_depth = torch.from_numpy(d[pts_2D[...,1], pts_2D[..., 0]])

#         valid = (pts_depth > 0.) and (torch.abs(im_depth- pts_depth) < threshold )
        
#         if valid.sum() < 1:
#             continue
#         pts_2D = pts_2D[valid]
        
        
#         f = downsample_sam_feature(f.transpose(2,0,1))
#         predictor.features = torch.from_numpy(f).to(predictor.device)[None, ...]
        
#         masks, scores, logits = predictor.predict(
#             point_coords=pts_2D.numpy(),
#             point_labels=input_label,
#             multimask_output=True,
#         )

        
#         for j, (mask, score) in enumerate(zip(masks, scores)):
#             output_file = os.path.join(output_root, f'test_{idx}.png')
#             plt.figure(figsize=(10,10))
#             plt.imshow(r)
#             show_mask(mask, plt.gca())
#             show_points(pts_2D, input_label, plt.gca())
#             plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#             plt.axis('off')
#             plt.savefig(output_file)
#             plt.close()
#     print(pts_3D)




def project_to_3d(pts, pose, intrinsics, depth):
    '''
    Args:
        pts: Nx2
        pose: 4x4
        intrinsics: fx, fy, cx, cy
        depth: HxW
    '''
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

def project_to_2d(pts, pose, intrinsics):
    fx, fy, cx, cy = intrinsics
    pose = torch.tensor(pose)
    pose = torch.inverse(pose)
    
    camera_pts = pts @ pose[:3, :3].T
    camera_pts = camera_pts + pose[:3,3]

    camera_x = camera_pts[..., 0] / camera_pts[..., -1] * fx + cx
    camera_y = camera_pts[..., 1] / camera_pts[..., -1] * fy + cy
    
    pts_depth = torch.norm(camera_pts, dim=-1)
    sign = torch.ones_like(pts_depth)
    sign[camera_pts[..., -1] < 0.] = -1.
    
    return torch.stack([camera_x, camera_y], dim = -1).to(torch.int), pts_depth * sign

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
    main()
    # get_3d_pts()
    # create_video()