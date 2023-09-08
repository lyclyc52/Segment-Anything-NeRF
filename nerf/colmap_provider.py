import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from .utils import get_rays
from .colmap_utils import *



def get_incoherent_mask(input_masks, sfact=2, keep_size=True):
    mask = input_masks.float()
    w = input_masks.shape[-1]
    h = input_masks.shape[-2]
    mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
    mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
    mask_residue = (mask - mask_recover).abs()
    mask_uncertain = F.interpolate(
        mask_residue, (h//sfact, w//sfact), mode='bilinear')
    mask_uncertain[mask_uncertain >= 0.01] = 1.
    
    if keep_size:
        mask_uncertain = F.interpolate(
            mask_uncertain, (h,w), mode='nearest')

    return mask_uncertain


def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def center_poses(poses, pts3d=None, enable_cam_center=False):
    
    def normalize(v):
        return v / (np.linalg.norm(v) + 1e-10)

    if pts3d is None or enable_cam_center:
        center = poses[:, :3, 3].mean(0)
    else:
        center = pts3d.mean(0)
        
    
    up = normalize(poses[:, :3, 1].mean(0)) # (3)
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    
    poses[:, :3, 3] -= center
    poses_centered = R @ poses # (N_images, 4, 4)

    if pts3d is not None:
        pts3d_centered = (pts3d - center) @ R[:3, :3].T
        # pts3d_centered = pts3d @ R[:3, :3].T - center
        return poses_centered, pts3d_centered

    return poses_centered


def visualize_poses(poses, size=0.05, bound=1, points=None):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    if points is not None:
        print('[visualize points]', points.shape, points.dtype, points.min(0), points.max(0))
        colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
        colors[:, 2] = 255 # blue
        colors[:, 3] = 30 # transparent
        objects.append(trimesh.PointCloud(points, colors))

    scene = trimesh.Scene(objects)
    scene.set_camera(distance=bound, center=[0, 0, 0])
    scene.show()


class ColmapDataset:
    def __init__(self, opt, device, type='train', n_test=24):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = opt.downscale
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        # self.offset = opt.offset # camera offset
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.root_path = opt.path # contains "colmap_sparse"

        self.training = self.type in ['train', 'all', 'trainval']

        # locate colmap dir
        candidate_paths = [
            os.path.join(self.root_path, "colmap_sparse", "0"),
            os.path.join(self.root_path, "sparse", "0"),
            os.path.join(self.root_path, "colmap"),
        ]
        
        self.colmap_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                self.colmap_path = path
                break

        if self.colmap_path is None:
            raise ValueError(f"Cannot find colmap sparse output under {self.root_path}, please run colmap first!")

        camdata = read_cameras_binary(os.path.join(self.colmap_path, 'cameras.bin'))

        # read image size (assume all images are of the same shape!)
        self.H = int(round(camdata[1].height / self.downscale))
        self.W = int(round(camdata[1].width / self.downscale))
        print(f'[INFO] ColmapDataset: image H = {self.H}, W = {self.W}')

        # read image paths
        imdata = read_images_binary(os.path.join(self.colmap_path, "images.bin"))
        imkeys = np.array(sorted(imdata.keys()))


        img_names = [os.path.basename(imdata[k].name) for k in imkeys]
        self.img_names = img_names


        img_folder = os.path.join(self.root_path, f"images_{self.downscale}")
        if not os.path.exists(img_folder):
            img_folder = os.path.join(self.root_path, "images")
        img_paths = np.array([os.path.join(img_folder, name) for name in img_names])
        
        
        if self.opt.with_mask:

            mask_folder = os.path.join(self.root_path, self.opt.mask_folder_name)
            mask_paths = np.array([os.path.join(mask_folder, name) for name in img_names])

        

        feature_folder = os.path.join(self.root_path, 'sam_features')
        feature_paths = np.array([os.path.join(feature_folder, name + '.npz') for name in img_names])

        # only keep existing images
        exist_mask = np.array([os.path.exists(f) for f in img_paths])
        print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
        imkeys = imkeys[exist_mask]
        img_paths = img_paths[exist_mask]
        feature_paths = feature_paths[exist_mask]

        # read intrinsics
        intrinsics = []
        for k in imkeys:
            cam = camdata[imdata[k].camera_id]
            if cam.model in ['SIMPLE_RADIAL', 'SIMPLE_PINHOLE']:
                fl_x = fl_y = cam.params[0] / self.downscale
                cx = cam.params[1] / self.downscale
                cy = cam.params[2] / self.downscale
            elif cam.model in ['PINHOLE', 'OPENCV']:
                fl_x = cam.params[0] / self.downscale
                fl_y = cam.params[1] / self.downscale
                cx = cam.params[2] / self.downscale
                cy = cam.params[3] / self.downscale
            else:
                raise ValueError(f"Unsupported colmap camera model: {cam.model}")
            intrinsics.append(np.array([fl_x, fl_y, cx, cy], dtype=np.float32))
        
        self.intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]

        # read poses
        poses = []
        for k in imkeys:
            P = np.eye(4, dtype=np.float64)
            P[:3, :3] = imdata[k].qvec2rotmat()
            P[:3, 3] = imdata[k].tvec
            poses.append(P)
        
        poses = np.linalg.inv(np.stack(poses, axis=0)) # [N, 4, 4]
        

        # read sparse points
        ptsdata = read_points3d_binary(os.path.join(self.colmap_path, "points3D.bin"))
        ptskeys = np.array(sorted(ptsdata.keys()))
        pts3d = np.array([ptsdata[k].xyz for k in ptskeys]) # [M, 3]
        self.ptserr = np.array([ptsdata[k].error for k in ptskeys]) # [M]
        self.mean_ptserr = np.mean(self.ptserr)

        # center pose
        self.poses, self.pts3d = center_poses(poses, pts3d, self.opt.enable_cam_center)
        print(f'[INFO] ColmapDataset: load poses {self.poses.shape}, points {self.pts3d.shape}')

        # rectify convention...
        self.poses[:, :3, 1:3] *= -1
        self.poses = self.poses[:, [1, 0, 2, 3], :]
        self.poses[:, 2] *= -1

        self.pts3d = self.pts3d[:, [1, 0, 2]]
        self.pts3d[:, 2] *= -1

        # auto-scale
        if self.scale == -1:
            self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).max()
            print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')

        self.poses[:, :3, 3] *= self.scale
        self.pts3d *= self.scale

        # use pts3d to estimate aabb
        # self.pts_aabb = np.concatenate([np.percentile(self.pts3d, 1, axis=0), np.percentile(self.pts3d, 99, axis=0)]) # [6]
        self.pts_aabb = np.concatenate([np.min(self.pts3d, axis=0), np.max(self.pts3d, axis=0)]) # [6]
        if np.abs(self.pts_aabb).max() > self.opt.bound:
            print(f'[WARN] ColmapDataset: estimated AABB {self.pts_aabb.tolist()} exceeds provided bound {self.opt.bound}! Consider improving --bound to make scene included in trainable region.')

        # process pts3d into sparse depth data.

        if self.type != 'test':
        
            self.cam_near_far = [] # always extract this infomation
            
            print(f'[INFO] extracting sparse depth info...')
            # map from colmap points3d dict key to dense array index
            pts_key_to_id = np.ones(ptskeys.max() + 1, dtype=np.int64) * len(ptskeys)
            pts_key_to_id[ptskeys] = np.arange(0, len(ptskeys))
            # loop imgs
            _mean_valid_sparse_depth = 0
            for i, k in enumerate(tqdm.tqdm(imkeys)):
                xys = imdata[k].xys
                xys = np.stack([xys[:, 1], xys[:, 0]], axis=-1) # invert x and y convention...
                pts = imdata[k].point3D_ids

                mask = (pts != -1) & (xys[:, 0] >= 0) & (xys[:, 0] < camdata[1].height) & (xys[:, 1] >= 0) & (xys[:, 1] < camdata[1].width)

                assert mask.any(), 'every image must contain sparse point'
                
                valid_ids = pts_key_to_id[pts[mask]]
                pts = self.pts3d[valid_ids] # points [M, 3]
                err = self.ptserr[valid_ids] # err [M]
                xys = xys[mask] # pixel coord [M, 2], float, original resolution!

                xys = np.round(xys / self.downscale).astype(np.int32) # downscale
                xys[:, 0] = xys[:, 0].clip(0, self.H - 1)
                xys[:, 1] = xys[:, 1].clip(0, self.W - 1)
                
                # calc the depth
                P = self.poses[i]
                depth = (P[:3, 3] - pts) @ P[:3, 2]

                # calc weight
                weight = 2 * np.exp(- (err / self.mean_ptserr) ** 2)

                _mean_valid_sparse_depth += depth.shape[0]

                # camera near far
                # self.cam_near_far.append([np.percentile(depth, 0.1), np.percentile(depth, 99.9)])
                self.cam_near_far.append([np.min(depth), np.max(depth)])

            print(f'[INFO] extracted {_mean_valid_sparse_depth / len(imkeys):.2f} valid sparse depth on average per image')

            self.cam_near_far = torch.from_numpy(np.array(self.cam_near_far, dtype=np.float32)) # [N, 2]

          
        else: # test time: no depth info
            self.cam_near_far = None


        # make split
        if self.type == 'test':
            
            poses = []

            if self.opt.camera_traj == 'circle':

                print(f'[INFO] use circular camera traj for testing.')
                
                # circle 360 pose
                # radius = np.linalg.norm(self.poses[:, :3, 3], axis=-1).mean(0)
                radius = 0.1
                theta = np.deg2rad(80)
                for i in range(100):
                    phi = np.deg2rad(i / 100 * 360)
                    center = np.array([
                        radius * np.sin(theta) * np.sin(phi),
                        radius * np.sin(theta) * np.cos(phi),
                        radius * np.cos(theta),
                    ])
                    # look at
                    def normalize(v):
                        return v / (np.linalg.norm(v) + 1e-10)
                    forward_v = normalize(center)
                    up_v = np.array([0, 0, 1])
                    right_v = normalize(np.cross(forward_v, up_v))
                    up_v = normalize(np.cross(right_v, forward_v))
                    # make pose
                    pose = np.eye(4)
                    pose[:3, :3] = np.stack((right_v, up_v, forward_v), axis=-1)
                    pose[:3, 3] = center
                    poses.append(pose)
                
                self.poses = np.stack(poses, axis=0)
            
            # choose some random poses, and interpolate between.
            else:

                fs = np.random.choice(len(self.poses), 5, replace=False)
                pose0 = self.poses[fs[0]]
                for i in range(1, len(fs)):
                    pose1 = self.poses[fs[i]]
                    rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                    slerp = Slerp([0, 1], rots)    
                    for i in range(n_test + 1):
                        ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, :3] = slerp(ratio).as_matrix()
                        pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                        poses.append(pose)
                    pose0 = pose1

                self.poses = np.stack(poses, axis=0)

            # fix intrinsics for test case
            self.intrinsics = self.intrinsics[[0]].repeat(self.poses.shape[0], 1)

            self.images = None
        
        else:
            all_ids = np.arange(len(img_paths))
            val_ids = all_ids[::16]
            if self.opt.val_all:
                val_ids = all_ids

            if self.type == 'train':
                train_ids = np.array([i for i in all_ids if i not in val_ids])
                self.poses = self.poses[train_ids]
                self.intrinsics = self.intrinsics[train_ids]
                img_paths = img_paths[train_ids]
                feature_paths = feature_paths[train_ids]
                if self.opt.with_mask:
                    mask_paths = mask_paths[train_ids]
                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[train_ids]
            elif self.type == 'val':
                self.poses = self.poses[val_ids]
                self.intrinsics = self.intrinsics[val_ids]
                img_paths = img_paths[val_ids]
                feature_paths = feature_paths[val_ids]
                if self.opt.with_mask:
                    mask_paths = mask_paths[val_ids]
                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[val_ids]
            # else: trainval use all.
            
            # read images
            if not self.opt.with_sam:
                self.images = []
                for f in tqdm.tqdm(img_paths, desc=f'Loading {self.type} image'):
                    image = cv2.imread(f, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                    # add support for the alpha channel as a mask.
                    if image.shape[-1] == 3: 
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    if image.shape[0] != self.H or image.shape[1] != self.W:
                        image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    self.images.append(image)
                self.images = np.stack(self.images, axis=0)
            else:
                self.images = None
                
            def validate_file(file_name):
                file_id = int(file_name[-15:-10])
                # return_value = file_id <= 30 or (file_id >=53 and file_id <=54 ) or (file_id >= 122 and file_id <=135) or \
                #     (file_id >= 171 and file_id <= 177) 
                
                # return_value = file_id <= 30 and file_id >=2
                # return return_value
                
                return True
            
            
            if self.opt.with_mask:
                self.masks = []
                self.valid_mask_index = []
        
                with open(os.path.join(mask_folder, 'valid.json')) as f:
                    valid_dict = json.load(f)
                    
                for idx in tqdm.tqdm(range(len(mask_paths)), desc=f'Loading {self.type} mask'):
                    f = mask_paths[idx]
                    
                    f = f.replace('.jpg', '_masks.npy').replace('.JPG', '_masks.npy').replace('.png', '_masks.npy').replace('.PNG', '_masks.npy')
                    mask = torch.from_numpy(np.load(f))
                    
                    
                    if mask.sum()>=10 and validate_file(f) and valid_dict[img_names[idx][:-4]]:
                        self.valid_mask_index.append(idx)
                        
                
                    self.masks.append(mask.to(int))
                self.masks = torch.stack(self.masks, axis=0)
                if self.opt.rgb_similarity_loss_weight > 0 or self.opt.incoherent_uncertainty_weight < 1:
                    self.incoherent_masks = get_incoherent_mask(self.masks.permute(0,3,1,2), sfact=2)                    
                    self.incoherent_masks = self.incoherent_masks.permute(0,2,3,1).to(torch.bool)[..., 0]
                    # np.save('debug/incoherent.npy', self.incoherent_masks.numpy())
                else:
                    self.incoherent_masks = None
                
                self.valid_mask_index = self.valid_mask_index[::8]
                # self.valid_mask_index = []
                # print(len(self.valid_mask_index))
                # exit()
                # self.valid_mask_index = np.array(self.valid_mask_index)
                
                

            else:
                self.masks = None
                self.incoherent_masks = None

        # view all poses.
        if self.opt.vis_pose:
            visualize_poses(self.poses, bound=self.opt.bound, points=self.pts3d)

        self.poses = torch.from_numpy(self.poses.astype(np.float32)) # [N, 4, 4]
        
        if self.opt.val_all:
            pose_dict = {}
            for i in range(len(self.img_names)):
                pose_dict[self.img_names[i][:-4]] = self.poses[i].numpy().tolist()
            with open(os.path.join(self.opt.workspace, 'pose_dir.json'), "w+") as f:
                json.dump(pose_dict, f, indent=4)


        if self.images is not None:
            self.images = torch.from_numpy(self.images.astype(np.uint8)) # [N, H, W, C]
            
        if self.preload:
            self.intrinsics = self.intrinsics.to(self.device)
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                self.images = self.images.to(self.device)
            if self.masks is not None:
                self.masks = self.masks.to(self.device)
            if self.incoherent_masks is not None:
                self.incoherent_masks = self.incoherent_masks.to(self.device)
            if self.cam_near_far is not None:
                self.cam_near_far = self.cam_near_far.to(self.device)

    def collate(self, index):
    
        num_rays = -1 # defaul, eval, test, train SAM use all rays
        if self.opt.with_mask:
            if index not in self.valid_mask_index:
                index= random.sample(self.valid_mask_index, 1)
                
                
        if self.training and not self.opt.with_sam :
            num_rays = self.opt.num_rays
            if self.opt.random_image_batch and not self.opt.with_mask:
                index = torch.randint(0, len(self.poses), size=(num_rays,), device=self.device if self.preload else 'cpu')

        H, W = self.H, self.W
        poses = self.poses[index] # [1/N, 4, 4]
        intrinsics = self.intrinsics[index] # [1/N, 4]

        if self.opt.with_sam:
            # augment poses
            if self.training:    
                H = W = self.opt.online_resolution
                fovy = 50 + 20 * random.random()
                focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
                intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)
                intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)
            # still use fixed pose, but change intrinsics
            else:
                H = W = self.opt.online_resolution
                fovy = 60
                focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
                intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)
                intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)

        if self.opt.with_mask:
            H = W = self.opt.online_resolution
            fovy = 60
            focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
            intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)
            intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)
    
        results = {'H': H, 'W': W}


        incoherent_mask = self.incoherent_masks[index] if self.incoherent_masks is not None else None
        # include_incoherent_region = torch.randint(0, 1, size=[1]) > 0
        include_incoherent_region = torch.tensor([0])
        

        rays = get_rays(poses, intrinsics, H, W, num_rays, device=self.device if self.preload else 'cpu', 
                        patch_size=self.opt.patch_size if self.opt.with_mask else 1, incoherent_mask=incoherent_mask,
                        include_incoherent_region=include_incoherent_region[0])


        img_names = [self.img_names[i] for i in index]
        names_without_suffix = []
        for n in img_names:
            names_without_suffix.append(n[:-4])
        results['img_names'] = names_without_suffix

        if self.images is not None:
            if num_rays != -1:
                images = self.images[index, rays['j'], rays['i']].float() / 255 # [N, 3/4]
            else:
                images = self.images[index].squeeze(0).float() / 255 # [H, W, 3/4]

            if self.training:
                C = self.images.shape[-1]
                images = images.view(-1, C)

            results['images'] = images.to(self.device)
        

        if self.masks is not None:
            if num_rays != -1:
                masks = self.masks[index, rays['j'], rays['i']] # [N, 1]
            else:
                masks = self.masks[index].squeeze(0) # [H, W, 1]

            if self.training:
                C = self.masks.shape[-1]
                masks = masks.view(-1, C)
            results['masks'] = masks.to(self.device)
            
        if self.incoherent_masks is not None:
            if num_rays != -1:
                incoherent_masks = self.incoherent_masks[index, rays['j'], rays['i']] # [N]
            else:
                incoherent_masks = self.incoherent_masks[index].squeeze(0) # [H, W]
                
                
            if self.training:
                incoherent_masks = masks.view(-1)
            results['incoherent_masks'] = incoherent_masks.to(self.device)
            
        
        if self.opt.enable_cam_near_far and self.cam_near_far is not None:
            cam_near_far = self.cam_near_far[index] # [1/N, 2]
            results['cam_near_far'] = cam_near_far.to(self.device)
        
        results['poses'] = poses.to(self.device)
        results['intrinsics'] = intrinsics.to(self.device)

        results['rays_o'] = rays['rays_o'].to(self.device)
        results['rays_d'] = rays['rays_d'].to(self.device)
        results['index'] = index.to(self.device) if torch.is_tensor(index) else index

        if self.opt.with_sam:
            scale = 16 * self.opt.online_resolution // 1024 
            rays_lr = get_rays(poses, intrinsics / scale, H // scale, W // scale, num_rays, device=self.device if self.preload else 'cpu')
            results['rays_o_lr'] = rays_lr['rays_o'].to(self.device)
            results['rays_d_lr'] = rays_lr['rays_d'].to(self.device)
            results['h'] = H // scale
            results['w'] = W // scale

        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader