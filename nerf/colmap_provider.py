import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh
from pathlib import Path

import torch
import torch.nn.functional as F

import time

from torch.utils.data import DataLoader

from .utils import get_rays, get_incoherent_mask
from .colmap_utils import *
import pyquaternion as pyquat





def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

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
        self.origin_num_local_sample = opt.num_local_sample
        self.origin_local_sample_patch_size = opt.local_sample_patch_size
        self.training = self.type in ['train', 'all', 'trainval']

        # locate colmap dir
        
        
        
        
        

        
        if self.opt.data_type != 'mip':
            if self.opt.data_type == '3dfront':
                with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                    transform = json.load(f)  
                if 'room_bbox' in transform:
                    room_bbox = np.array(transform['room_bbox'])
                    self.offset = -(room_bbox[0] + room_bbox[1]) * 0.5 * self.scale

                self.H = int(transform["h"])
                self.W = int(transform["w"])
                
                img_folder = os.path.join(self.root_path, f"images_{self.downscale}")
                if not os.path.exists(img_folder):
                    img_folder = os.path.join(self.root_path, "images")
                img_paths = []
                poses = []
                intrinsics = []
                cam_near_far = None
                pose = [] # [4, 4]
                
                for frame in transform['frames']:
                    img_paths.append(os.path.join(self.root_path, frame['file_path']))
                    pose = nerf_matrix_to_ngp(np.array(frame['transform_matrix'],  dtype=np.float32), scale=1) # this may not be necessary
                    pose[:, 1:3] = -pose[:, 1:3] 
                    poses.append(pose)
                    intrinsics.append(np.array([transform["fl_x"], transform["fl_y"], transform["cx"], transform["cy"]], dtype=np.float32))
                    # cam_near_far.append(np.array([0.2, 12]))
                img_names = [os.path.basename(img) for img in img_paths]
                self.img_names = np.array(img_names)
                img_paths = np.array(img_paths)
                
                self.intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]

                    
                self.poses = np.stack(poses)
                # change the direction of yz to make sure the camera looks at z-
                self.poses[:, :3, 1:3] *= -1

                self.pts3d = self.poses[:, :3, 3] # [M, 3]
                self.poses, self.pts3d = center_poses(self.poses, self.pts3d, self.opt.enable_cam_center)
                
                if self.scale == -1:
                    self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).max()
                    # self.scale = 0.33
                    print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')

                self.poses[:, :3, 3] *= self.scale
                self.pts3d *= self.scale

                # use pts3d to estimate aabb
                # self.pts_aabb = np.concatenate([np.percentile(self.pts3d, 1, axis=0), np.percentile(self.pts3d, 99, axis=0)]) # [6]
                self.pts_aabb = np.concatenate([np.min(self.pts3d, axis=0), np.max(self.pts3d, axis=0)]) # [6]
                if np.abs(self.pts_aabb).max() > self.opt.bound:
                    print(f'[WARN] ColmapDataset: estimated AABB {self.pts_aabb.tolist()} exceeds provided bound {self.opt.bound}! Consider improving --bound to make scene included in trainable region.')

                
                feature_folder = os.path.join(self.root_path, 'sam_features')
                feature_paths = np.array([os.path.join(feature_folder, name + '.npz') for name in img_names])
                
                exist_mask = np.array([os.path.exists(f) for f in img_paths])
                print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
                img_paths = img_paths[exist_mask]
                feature_paths = feature_paths[exist_mask]
                self.poses = self.poses[exist_mask]
                self.intrinsics= self.intrinsics[exist_mask]
                if self.opt.with_mask:
                    mask_folder = os.path.join(self.root_path, self.opt.mask_folder_name)
                    mask_paths = np.array([os.path.join(mask_folder, name) for name in img_names])

            elif self.opt.data_type == 'llff':

                with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                    transform = json.load(f)



                self.H = int(transform["h"])
                self.W = int(transform["w"])
                
                img_folder = os.path.join(self.root_path, f"images_{self.downscale}")
                if not os.path.exists(img_folder):
                    img_folder = os.path.join(self.root_path, "images")
                img_paths = []
                poses = []
                intrinsics = []
                cam_near_far = None
                pose = [] # [4, 4]
                
                for frame in transform['frames']:
                    img_paths.append(os.path.join(self.root_path, frame['file_path']))
                    pose = nerf_matrix_to_ngp(np.array(frame['transform_matrix'],  dtype=np.float32), scale=1)
                    pose[:, 1:3] = -pose[:, 1:3] 
                    poses.append(pose)
                    intrinsics.append(np.array([transform["fl_x"], transform["fl_y"], transform["cx"], transform["cy"]], dtype=np.float32))
                    # cam_near_far.append(np.array([0.2, 12]))
                img_names = [os.path.basename(img) for img in img_paths]
                self.img_names = np.array(img_names)
                img_paths = np.array(img_paths)
                
                self.intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]               
                self.poses  = np.stack(poses)
                # change the direction of yz to make sure the camera looks at z-
                self.poses[:, :3, 1:3] *= -1
                
                self.pts3d = self.poses[:, :3, 3] # [M, 3]

                if self.scale == -1:
                    # self.scale = 1 / self.poses[:, :3, 3].reshape(-1).max()
                    self.scale = 0.33
                    print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')
                self.poses[:, :3, 3] *= self.scale
                self.pts3d *= self.scale

                # use pts3d to estimate aabb
                # self.pts_aabb = np.concatenate([np.percentile(self.pts3d, 1, axis=0), np.percentile(self.pts3d, 99, axis=0)]) # [6]
                self.pts_aabb = np.concatenate([np.min(self.pts3d, axis=0), np.max(self.pts3d, axis=0)]) # [6]

                if np.abs(self.pts_aabb).max() > self.opt.bound:
                    print(f'[WARN] ColmapDataset: estimated AABB {self.pts_aabb.tolist()} exceeds provided bound {self.opt.bound}! Consider improving --bound to make scene included in trainable region.')

                # self.poses = self.poses[:, [1, 0, 2, 3], :]
                # self.cam_near_far = torch.from_numpy(np.stack(cam_near_far))
                
                if self.opt.with_mask:
                    mask_folder = os.path.join(self.root_path, self.opt.mask_folder_name)
                    mask_paths = np.array([os.path.join(mask_folder, name) for name in img_names])

                feature_folder = os.path.join(self.root_path, 'sam_features')
                feature_paths = np.array([os.path.join(feature_folder, name + '.npz') for name in img_names])
                
                exist_mask = np.array([os.path.exists(f) for f in img_paths])
                print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
                img_paths = img_paths[exist_mask]
                feature_paths = feature_paths[exist_mask]
                self.poses = self.poses[exist_mask]
                self.intrinsics= self.intrinsics[exist_mask]
                
            elif self.opt.data_type == 'lift':
                
                img_folder = os.path.join(self.root_path, f"images_{self.downscale}")
                if not os.path.exists(img_folder):
                    img_folder = os.path.join(self.root_path, "images")

                img_names = os.listdir(img_folder)
                img_names.sort()
                self.img_names = np.array(img_names)
                img_paths = np.array([os.path.join(img_folder, name) for name in img_names])
                poses = []
                intrinsics = []

                self.H, self.W = cv2.imread(img_paths[0]).shape[:2]
    
                pose_root = os.path.join(self.root_path, 'metadata.json')
                if os.path.isfile(pose_root):
                    with open(pose_root) as f:
                        meta = json.load(f)
                    global_intr = np.array(meta['camera']['K'])
                    global_intr[0] *= self.W 
                    global_intr[1] *= self.H
                    global_intr = np.array([global_intr[0,0], global_intr[1,1], global_intr[0,-1], global_intr[1,-1]])
                    # global_intr = np.array([meta['camera']['focal_length'], meta['camera']['focal_length'], global_intr[0,-1], global_intr[1,-1]])

                    global_intr = np.abs(global_intr, dtype=np.float32)

                    
                    for i in range(len(meta["camera"]["positions"])):
                        pose = np.eye(4)
                        t = np.array(meta["camera"]["positions"][i])
                        q = np.array(meta["camera"]["quaternions"][i])
                        rot = pyquat.Quaternion(*q).rotation_matrix
                        pose[:3, :3] = rot
                        pose[:3, 3] = t

                        poses.append(pose)
                        intrinsics.append(global_intr)
                else:
                    pose_root = os.path.join(self.root_path, 'pose')
                    intri_file = os.path.join(self.root_path, 'intrinsic', 'intrinsic_color.txt')
                    global_intr = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(intri_file).read_text().splitlines() if x != ''])
                    global_intr = np.array([global_intr[0,0], global_intr[1,1], global_intr[0,-2], global_intr[1,-2]], dtype=np.float32)
                    # global_intr = np.array([meta['camera']['focal_length'], meta['camera']['focal_length'], global_intr[0,-1], global_intr[1,-1]])
                    for name in img_names:
                        pose_name = os.path.join(pose_root, name[:-3] + 'txt')
                        pose = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(pose_name).read_text().splitlines() if x != ''])
                        pose[:, 1:3] = -pose[:, 1:3] 
                        poses.append(pose)
                        intrinsics.append(global_intr)
                # poses.append(np.linalg.inv( nerf_matrix_to_ngp(np.array(frame['transform_matrix'],  dtype=np.float32), scale=self.scale, offset=self.offset)))
                # poses.append(nerf_matrix_to_ngp(np.array(frame['transform_matrix'],  dtype=np.float32), scale=self.scale, offset=self.offset))
                self.poses  = np.stack(poses, axis=0)
                self.pts3d = self.poses[:, :3, 3]
               
                self.intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]


                self.poses, self.pts3d = center_poses(self.poses, self.pts3d, self.opt.enable_cam_center)


                if self.scale == -1:
                    self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).max()
                    print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')

                self.poses[:, :3, 3] *= self.scale

                self.pts3d *= self.scale

                # use pts3d to estimate aabb
                # self.pts_aabb = np.concatenate([np.percentile(self.pts3d, 1, axis=0), np.percentile(self.pts3d, 99, axis=0)]) # [6]
                self.pts_aabb = np.concatenate([np.min(self.poses[:, :3, 3], axis=0), np.max(self.poses[:, :3, 3], axis=0)]) # [6]

                if np.abs(self.pts_aabb).max() > self.opt.bound:
                    print(f'[WARN] ColmapDataset: estimated AABB {self.pts_aabb.tolist()} exceeds provided bound {self.opt.bound}! Consider improving --bound to make scene included in trainable region.')


                feature_folder = os.path.join(self.root_path, 'sam_features')
                feature_paths = np.array([os.path.join(feature_folder, name + '.npz') for name in img_names])
                
                
                if self.opt.with_mask:
                    mask_folder = os.path.join(self.root_path, self.opt.mask_folder_name)
                    mask_paths = np.array([os.path.join(mask_folder, name) for name in img_names])

                exist_mask = np.array([os.path.exists(f) for f in img_paths])
                print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
                img_paths = img_paths[exist_mask]
                feature_paths = feature_paths[exist_mask]
                self.poses = self.poses[exist_mask]
                self.intrinsics= self.intrinsics[exist_mask]
            elif self.opt.data_type == 'pano':
                                   
                img_folder = os.path.join(self.root_path, f"images_{self.downscale}")
                if not os.path.exists(img_folder):
                    img_folder = os.path.join(self.root_path, "images")

                img_names = os.listdir(img_folder)
                img_names.sort()
                self.img_names = np.array(img_names)
                
                img_paths = np.array([os.path.join(img_folder, name) for name in img_names])
                poses = []
                intrinsics = []

                self.H, self.W = cv2.imread(img_paths[0]).shape[:2]             
                pose_root = os.path.join(self.root_path, 'pose')
                
                intri_file = os.path.join(self.root_path, 'intrinsic', 'intrinsic_color.txt')
                global_intr = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(intri_file).read_text().splitlines() if x != ''])
                global_intr = np.array([global_intr[0,0], global_intr[1,1], global_intr[0,-2], global_intr[1,-2]], dtype=np.float32)
                # global_intr = np.array([meta['camera']['focal_length'], meta['camera']['focal_length'], global_intr[0,-1], global_intr[1,-1]])


                
                for name in img_names:
                    pose_name = os.path.join(pose_root, name[:-3] + 'txt')
                    pose = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(pose_name).read_text().splitlines() if x != ''])
                    # pose = nerf_matrix_to_ngp(np.array(pose,  dtype=np.float32), scale=1)
                    pose[:, 1:3] = -pose[:, 1:3] 
                    poses.append(pose)
                    intrinsics.append(global_intr)
                self.poses  = np.stack(poses, axis=0)


                self.pts3d = self.poses[:, :3, 3]
               
         
                self.intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]
                self.poses, self.pts3d = center_poses(self.poses, self.pts3d, self.opt.enable_cam_center)


                if self.scale == -1:
                    self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).max()
                    print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')

                self.poses[:, :3, 3] *= self.scale

                self.pts3d *= self.scale

                # use pts3d to estimate aabb
                # self.pts_aabb = np.concatenate([np.percentile(self.pts3d, 1, axis=0), np.percentile(self.pts3d, 99, axis=0)]) # [6]
                self.pts_aabb = np.concatenate([np.min(self.poses[:, :3, 3], axis=0), np.max(self.poses[:, :3, 3], axis=0)]) # [6]

                if np.abs(self.pts_aabb).max() > self.opt.bound:
                    print(f'[WARN] ColmapDataset: estimated AABB {self.pts_aabb.tolist()} exceeds provided bound {self.opt.bound}! Consider improving --bound to make scene included in trainable region.')

                # self.poses = self.poses[:, [1, 0, 2, 3], :]
                # self.cam_near_far = torch.from_numpy(np.stack(cam_near_far))
                
                feature_folder = os.path.join(self.root_path, 'sam_features')
                feature_paths = np.array([os.path.join(feature_folder, name + '.npz') for name in img_names])
                
                exist_mask = np.array([os.path.exists(f) for f in img_paths])
                print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
                img_paths = img_paths[exist_mask]
                feature_paths = feature_paths[exist_mask]
                self.poses = self.poses[exist_mask]
                self.intrinsics= self.intrinsics[exist_mask]  
                
        else:
            self.colmap_path = None
            candidate_paths = [
                os.path.join(self.root_path, "colmap_sparse", "0"),
                os.path.join(self.root_path, "sparse", "0"),
                os.path.join(self.root_path, "colmap"),
            ]
            for path in candidate_paths:
                if os.path.exists(path):
                    self.colmap_path = path
                    break
                
            if self.colmap_path == None:
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
            self.img_names = np.array(img_names)


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
            
            self.poses = poses
            # # center pose
            self.poses, self.pts3d = center_poses(poses, pts3d, self.opt.enable_cam_center)
            print(f'[INFO] ColmapDataset: load poses {self.poses.shape}, points {self.pts3d.shape}')

            # rectify convention...
            self.poses[:, :3, 1:3] *= -1
            self.poses = self.poses[:, [1, 0, 2, 3], :]
            self.poses[:, 2] *= -1



            # print(np.linalg.inv((nerf_matrix_to_ngp(temp))))
            self.pts3d = self.pts3d[:, [1, 0, 2]]
            self.pts3d[:, 2] *= -1

            # auto-scale
            if self.scale == -1:
                self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).max()
                print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')

            self.poses[:, :3, 3] *= self.scale
            self.pts3d *= self.scale
            
            
            self.cam_near_far = []
            
            # use pts3d to estimate aabb
            # self.pts_aabb = np.concatenate([np.percentile(self.pts3d, 1, axis=0), np.percentile(self.pts3d, 99, axis=0)]) # [6]
            self.pts_aabb = np.concatenate([np.min(self.pts3d, axis=0), np.max(self.pts3d, axis=0)]) # [6]
            if np.abs(self.pts_aabb).max() > self.opt.bound:
                print(f'[WARN] ColmapDataset: estimated AABB {self.pts_aabb.tolist()} exceeds provided bound {self.opt.bound}! Consider improving --bound to make scene included in trainable region.')

            # process pts3d into sparse depth data.

        if self.type != 'test' and self.opt.data_type == 'mip':
        
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
                # weight = 2 * np.exp(- (err / self.mean_ptserr) ** 2)

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
            self.masks = None
            self.gt_incoherent_masks = None
            self.error_map = None
        
        else:
            all_ids = np.arange(len(img_paths))
            
            with open(os.path.join(self.root_path, 'data_split.json')) as f:
                data_split = json.load(f)
            # train_ids = [id for id in all_ids if self.img_names[id] in data_split['train']]

            if self.opt.val_type == 'default':
                val_ids = all_ids[::16]
            elif self.opt.val_type == 'val_all':
                val_ids = all_ids
            elif self.opt.val_type == 'val_split':
                if self.opt.with_mask:
                    with open(os.path.join(mask_folder, 'data_split.json')) as f:
                        data_split = json.load(f)
                        
                    val_ids = [idx for idx in all_ids if self.img_names[idx][:-4] in data_split['test']]
                else:
                    val_ids = [idx for idx in all_ids if self.img_names[idx] in data_split['test']]

            # val_ids = all_ids[::16]
            if self.type == 'train':
                train_ids = np.array([i for i in all_ids if i not in val_ids])
                self.poses = self.poses[train_ids]
                self.intrinsics = self.intrinsics[train_ids]
                img_paths = img_paths[train_ids]
                feature_paths = feature_paths[train_ids]
                if self.opt.with_mask:
                    mask_paths = mask_paths[train_ids]
                    self.img_names = self.img_names[train_ids]
                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[train_ids]
            elif self.type == 'val':
                self.poses = self.poses[val_ids]
                self.intrinsics = self.intrinsics[val_ids]
                img_paths = img_paths[val_ids]
                feature_paths = feature_paths[val_ids]
                if self.opt.with_mask:
                    mask_paths = mask_paths[val_ids]
                    self.img_names = self.img_names[val_ids]
                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[val_ids]
            # elif self.type == 'test':
            #     for idx in range(20):
                    
            #         f0, f1 = np.random.choice(frames, 2, replace=False)
                    # pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
                    # pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
                    # rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                    # slerp = Slerp([0, 1], rots)

                    # self.poses = []
                    # self.images = None
                    # for i in range(n_test + 1):
                    #     ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                    #     pose = np.eye(4, dtype=np.float32)
                    #     pose[:3, :3] = slerp(ratio).as_matrix()
                    #     pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    #     self.poses.append(pose)
                    # else: trainval use all.
            
            # read images

            if not self.opt.with_sam and not self.opt.with_mask:
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
                # file_id = int(file_name[-15:-10])
                # return_value = file_id <= 30 or (file_id >=53 and file_id <=54 ) or (file_id >= 122 and file_id <=135) or \
                #                 (file_id >= 171 and file_id <= 177)
                                 
                # return return_value
                
                return True

            if self.opt.with_mask:
                self.masks = []
                self.valid_mask_index_list = []
                if self.training:
                    with open(os.path.join(mask_folder, 'valid.json')) as f:
                        valid_dict = json.load(f)
                    
                for idx in tqdm.tqdm(range(len(mask_paths)), desc=f'Loading {self.type} mask'):
                    f = mask_paths[idx]
                    f = f.replace('.jpg', '_masks.npy').replace('.JPG', '_masks.npy').replace('.png', '_masks.npy').replace('.PNG', '_masks.npy')
                    if os.path.isfile(f):
                        mask = torch.from_numpy(np.load(f))
                    else:
                        mask = torch.zeros([512, 512])
                    
                    
                    # mask_list = ['DSC07956', 'DSC07957', 'DSC07958', 'DSC07959', 'DSC07960', 'DSC07961', 'DSC07963',
                    #              'DSC07964', 'DSC07969', 'DSC07970', 'DSC07971', 'DSC07982', 'DSC07984',
                    #              'DSC07985', 'DSC07987',
                    #              'DSC08029', 'DSC08032', 'DSC08033', 'DSC08034', 'DSC08035'
                    #              ]
                    # print(valid_dict)
                    # exit()
                    if self.training:
                        if mask.sum()>=10 and validate_file(f) and valid_dict[self.img_names[idx][:-4]]:
                            self.valid_mask_index_list.append(idx)
                            
                        # if self.img_names[idx][:-4] in mask_list:
                        #     self.valid_mask_index_list.append(idx)    

                
                    self.masks.append(mask.to(int))
                self.masks = torch.stack(self.masks, axis=0)
                if len(self.masks.shape) != 4:
                    self.masks = self.masks[..., None]

                
                self.origin_H, self.origin_W = self.masks.shape[1], self.masks.shape[2]
                self.H, self.W = self.origin_H, self.origin_W
                if self.opt.rgb_similarity_loss_weight > 0 or self.opt.incoherent_uncertainty_weight < 1:
                    self.gt_incoherent_masks = get_incoherent_mask(self.masks.permute(0,3,1,2), sfact=2)                    
                    self.gt_incoherent_masks = self.gt_incoherent_masks.permute(0,2,3,1).to(torch.bool)[..., 0]
                    self.gt_incoherent_masks = self.gt_incoherent_masks.view(-1, self.H*self.W)
                else:
                    self.gt_incoherent_masks = None
                
                if self.training:
                    old_valid_mask_index_list = np.array(self.valid_mask_index_list)
                    if len(old_valid_mask_index_list) // 5 <= 25:
                        add_sample = np.random.choice(self.valid_mask_index_list, 25)
                        self.valid_mask_index_list = add_sample
                    else:
                        self.valid_mask_index_list = old_valid_mask_index_list[::3]
                        
                        
                    # sample_num = len(self.valid_mask_index_list)
                    # if sample_num < 25 and self.type != 'llff':
                    #     add_sample = np.random.choice(self.valid_mask_index_list, 25 - sample_num)
                    #     self.valid_mask_index_list = np.concatenate([self.valid_mask_index_list, add_sample])
                    
                        
                    self.valid_mask_index = torch.tensor(self.valid_mask_index_list).to(torch.int)
                
                    self.poses = self.poses[self.valid_mask_index]
                    self.masks = self.masks[self.valid_mask_index]
                    # for m in range(self.masks.shape[0]):
                    #     print(self.masks[m].sum())
                    # exit()
                    self.confident_masks = self.masks.clone()
                    self.img_names = [self.img_names[idx] for idx in self.valid_mask_index_list]
                    if self.gt_incoherent_masks is not None:
                        self.gt_incoherent_masks = self.gt_incoherent_masks[self.valid_mask_index]
                    if self.cam_near_far is not None:
                        self.cam_near_far = self.cam_near_far[self.valid_mask_index]
                    np.save('./debug/masks.npy', self.masks.numpy())
                    # exit()
                    if self.opt.error_map:
                        self.error_map = torch.ones([self.masks.shape[0], self.opt.error_map_size * self.opt.error_map_size], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
                    else:
                        self.error_map = None
                else:
                    self.error_map = None
                # self.valid_mask_index = []
                # print(len(self.valid_mask_index))
                # exit()
                # self.valid_mask_index = np.array(self.valid_mask_index)

            else:
                self.masks = None
                self.confident_masks = None
                self.gt_incoherent_masks = None
                self.error_map = None
        
        # view all poses.
        if self.opt.vis_pose:
            visualize_poses(self.poses, bound=self.opt.bound, points=self.pts3d)

        self.poses = torch.from_numpy(self.poses.astype(np.float32)) # [N, 4, 4]
        
        if self.opt.val_type == 'val_all' and self.type == 'val':
            pose_dict = {}
            for i in range(len(self.img_names)):
                pose_dict[self.img_names[i][:-4]] = self.poses[i].numpy().tolist()
            with open(os.path.join(self.opt.workspace, 'validation', 'pose_dir.json'), "w+") as f:
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
                if self.training:
                    self.valid_mask_index = self.valid_mask_index.to(self.device)
                    if self.confident_masks is not None:
                        self.confident_masks = self.confident_masks.to(self.device)
            if self.gt_incoherent_masks is not None:
                self.gt_incoherent_masks = self.gt_incoherent_masks.to(self.device)
            
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)
            if self.cam_near_far is not None:
                self.cam_near_far = self.cam_near_far.to(self.device)
                
        if self.opt.use_dynamic_incoherent:
            self.incoherent_mask_size = int(self.H / self.opt.incoherent_downsample_scale)
            self.incoherent_masks = None
        else:
            self.incoherent_mask_size = self.H
            self.incoherent_masks = self.gt_incoherent_masks
            
        
    
    
    def collate_mask(self, index):
        num_rays = -1
        index = [index]
    
        H = W = self.incoherent_mask_size 
        fovy = 60
        focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
        intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)
        results = {'H': H, 'W': W}
        poses = self.poses[index] # [1/N, 4, 4]
        rays = get_rays(poses, intrinsics, H, W, num_rays, device=self.device if self.preload else 'cpu')
        
                
        if self.opt.enable_cam_near_far and self.cam_near_far is not None:
            cam_near_far = self.cam_near_far[index] # [1/N, 2]
            results['cam_near_far'] = cam_near_far.to(self.device)

        
        results['poses'] = poses.to(self.device)
        results['intrinsics'] = intrinsics.to(self.device)

        results['rays_o'] = rays['rays_o'].to(self.device)
        results['rays_d'] = rays['rays_d'].to(self.device)
        results['index'] = index.to(self.device) if torch.is_tensor(index) else index
        return results
    
    
    def collate(self, index):
        num_rays = -1 # defaul, eval, test, train SAM use all rays
        random_sample = False
        

        if self.training and self.opt.use_multi_res and self.global_step > self.opt.rgb_similarity_iter:
            multi_res_step = self.global_step - self.opt.rgb_similarity_iter
            if (multi_res_step - 1 % self.opt.multi_res_update_iter) == 0:
                downsample_level = multi_res_step // self.opt.multi_res_update_iter
                downsample_scale = pow(2, max(0, self.opt.max_multi_res_level - downsample_level))
            
                self.H, self.W = self.origin_H // downsample_scale, self.origin_W // downsample_scale
                if self.masks is not None:
                    self.masks = F.interpolate(self.confident_masks.float().permute(0,3,1,2), (self.H, self.W), mode='bilinear')
                    self.masks = self.masks.permute(0,2,3,1).long()
                if self.incoherent_masks is not None:
                    self.incoherent_masks = F.interpolate(self.incoherent_masks.float().permute(0,3,1,2), (self.H, self.W), mode='bilinear')
                    self.incoherent_mask_size = self.H
                    self.incoherent_masks = self.incoherent_masks.permute(0,2,3,1).long()
                if self.error_map is not None:
                    self.error_map = self.error_map.view(-1, self.opt.error_map_size, self.opt.error_map_size)
                    self.error_map = F.interpolate(self.error_map[:, None, ...], (self.H, self.W), mode='bilinear')
                    self.error_map = self.error_map[:, 0, ...]
                    self.opt.error_map_size = self.H
                    self.error_map = self.error_map.view(-1, self.opt.error_map_size * self.opt.error_map_size)

                self.opt.num_local_sample = (self.origin_num_local_sample // downsample_scale) // downsample_scale
                self.opt.local_sample_patch_size = self.origin_local_sample_patch_size // downsample_scale


        # Enable random sampling after a few epoch
        if self.training and (self.global_step > self.opt.rgb_similarity_iter or self.global_step / len(self.poses) > 3):
            self.opt.random_image_batch = True
            
            
        if self.training and not self.opt.with_sam:
            num_rays = self.opt.num_rays
            if self.opt.random_image_batch:
                if self.opt.with_mask:
                    if self.global_step <= self.opt.rgb_similarity_iter or self.opt.rgb_similarity_iter < 0 or self.opt.patch_size <= 1:
                        index = torch.randint(0, len(self.poses), size=(num_rays,), device=self.device if self.preload else 'cpu')
                        random_sample = True   
                else:
                    index = torch.randint(0, len(self.poses), size=(num_rays,), device=self.device if self.preload else 'cpu')
                    random_sample = True
        


        H, W = self.H, self.W

        poses = self.poses[index] # [1/N, 4, 4]
        intrinsics = self.intrinsics[index] # [1/N, 4]
        

        if self.opt.with_sam and not self.opt.with_mask:
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
            # H = W = self.opt.online_resolution 
            fovy = 60
            focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
            intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)
            intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)
    
    

        results = {'H': H, 'W': W}

        if not random_sample and self.opt.patch_size > 1:
            incoherent_mask = self.gt_incoherent_masks[index] if self.gt_incoherent_masks is not None else None
            # include_incoherent_region = torch.randint(0, 1, size=[1]) > 0
            include_incoherent_region = True
        else:
            incoherent_mask = None
            include_incoherent_region = False
        
        error_map = None if self.error_map is None else self.error_map[index]
 
        # This part is for global sampling 
        if self.opt.error_map:
            rays = get_rays(poses, intrinsics, H, W, num_rays, device=self.device if self.preload else 'cpu', 
                            patch_size=self.opt.patch_size if self.opt.with_mask else 1, incoherent_mask=error_map,
                            include_incoherent_region=True, incoherent_mask_size=self.opt.error_map_size, 
                            random_sample=random_sample)
        else:
            rays = get_rays(poses, intrinsics, H, W, num_rays, device=self.device if self.preload else 'cpu', 
                            patch_size=self.opt.patch_size if self.opt.with_mask else 1, incoherent_mask=incoherent_mask,
                            include_incoherent_region=include_incoherent_region, incoherent_mask_size=self.H,
                            random_sample=random_sample)
            
        # This part is for sampling in local sense
        if self.opt.mixed_sampling and self.training and self.global_step > self.opt.rgb_similarity_iter:

            local_indices = torch.randint(0, len(self.poses), size=(self.opt.num_local_sample,), device=self.device if self.preload else 'cpu')
            local_indices_expand = local_indices[..., None].expand(-1, self.opt.local_sample_patch_size * self.opt.local_sample_patch_size)
            local_indices_expand = local_indices_expand.reshape(-1)

            local_poses = self.poses[local_indices_expand]

                

            
            if self.opt.error_map:
                local_error_map = None if self.error_map is None else self.error_map[local_indices]
                local_rays = get_rays(local_poses, intrinsics, H, W, 1, device=self.device if self.preload else 'cpu', 
                                patch_size=self.opt.local_sample_patch_size, incoherent_mask=local_error_map,
                                include_incoherent_region=True, incoherent_mask_size=self.opt.error_map_size, 
                                random_sample=False)
            else:
                local_incoherent_mask = self.incoherent_masks[local_indices] if self.incoherent_masks is not None else None
                
                local_rays = get_rays(local_poses, intrinsics, H, W, 1, device=self.device if self.preload else 'cpu', 
                                patch_size=self.opt.local_sample_patch_size, incoherent_mask=local_incoherent_mask,
                                include_incoherent_region=True, incoherent_mask_size=self.H,
                                random_sample=False)

        # print(self.images)
        
        if not self.training and self.type == 'val':
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
                masks = self.masks[index, rays['j'], rays['i']]
                if self.opt.mixed_sampling and self.global_step > self.opt.rgb_similarity_iter:
                    local_masks = self.masks[local_indices_expand, local_rays['j'], local_rays['i']]
                    
                    masks = torch.cat([masks, local_masks], 0)
            else:
                masks = self.masks[index].squeeze(0) # [H, W, 1]

            if self.training:
                C = self.masks.shape[-1]
                masks = masks.view(-1, C)
            results['masks'] = masks.to(self.device)

        
        if self.incoherent_masks is not None:
            if num_rays != -1:
                incoherent_scale = self.incoherent_mask_size / self.H
                
                r_j = (rays['j'] * incoherent_scale).long()
                r_i = (rays['i'] * incoherent_scale).long()
                
                incoherent_masks = self.incoherent_masks[index, r_j* self.incoherent_mask_size + r_i] # [N]
                if self.opt.mixed_sampling and self.global_step > self.opt.rgb_similarity_iter:
                    local_r_j = (local_rays['j'] * incoherent_scale).long()
                    local_r_i = (local_rays['i'] * incoherent_scale).long()
                    local_incoherent_masks = self.incoherent_masks[local_indices_expand, local_r_j* self.incoherent_mask_size + local_r_i]
                    incoherent_masks = torch.cat([incoherent_masks, local_incoherent_masks], 0)
            else:
                incoherent_masks = self.incoherent_masks[index].squeeze(0) # [H, W]
                
                
            if self.training:
                incoherent_masks = incoherent_masks.view(-1)
            results['incoherent_masks'] = incoherent_masks.to(self.device)


        if self.error_map is not None:
            if num_rays != -1:
                error_scale = self.opt.error_map_size / self.H
                
                r_j = (rays['j'] * error_scale).long()
                r_i = (rays['i'] * error_scale).long()
                
                error_maps = self.error_map[index, r_j* self.opt.error_map_size + r_i] # [N]
                if self.opt.mixed_sampling and self.global_step > self.opt.rgb_similarity_iter:
                    
                    local_r_j = (local_rays['j'] * error_scale).long()
                    local_r_i = (local_rays['i'] * error_scale).long()
                    local_error_maps = self.error_map[local_indices_expand, local_r_j* self.opt.error_map_size + local_r_i]
                    error_maps = torch.cat([error_maps, local_error_maps], 0)
            else:
                error_maps = self.error_maps[index].squeeze(0) # [H, W]
                
                
            if self.training:
                error_maps = error_maps.view(-1)
            results['error_maps'] = error_maps.to(self.device)
        
        if self.opt.enable_cam_near_far and self.cam_near_far is not None:
            cam_near_far = self.cam_near_far[index] # [1/N, 2]
            results['cam_near_far'] = cam_near_far.to(self.device)
            if self.opt.mixed_sampling and self.global_step > self.opt.rgb_similarity_iter:
                local_cam_near_far = self.cam_near_far[local_indices_expand].to(self.device)
                results['cam_near_far'] = torch.cat([results['cam_near_far'], local_cam_near_far], 0)
            
        
        results['poses'] = poses.to(self.device)
        results['intrinsics'] = intrinsics.to(self.device)

        results['rays_o'] = rays['rays_o'].to(self.device)
        results['rays_d'] = rays['rays_d'].to(self.device)
        results['index'] = index.to(self.device) if torch.is_tensor(index) else index
        if self.opt.error_map and self.training:
            results['inds_coarse'] = rays['inds_coarse']

        if self.opt.mixed_sampling and self.training and self.global_step > self.opt.rgb_similarity_iter:
            results['poses'] = torch.cat([results['poses'], local_poses.to(self.device)], 0)
            results['rays_o'] = torch.cat([results['rays_o'], local_rays['rays_o'].to(self.device)], 0)
            results['rays_d'] = torch.cat([results['rays_d'], local_rays['rays_d'].to(self.device)], 0)
            # results['index'] = torch.cat([results['index'], local_indices_expand], 0)
 
            # if self.opt.error_map:
            #     results['inds_coarse'] = torch.cat([results['inds_coarse'], local_rays['inds_coarse']], 0)


        if self.opt.with_sam and not self.opt.with_mask:
            scale = 16 * self.opt.online_resolution // 1024 
            rays_lr = get_rays(poses, intrinsics / scale, H // scale, W // scale, num_rays, device=self.device if self.preload else 'cpu')
            results['rays_o_lr'] = rays_lr['rays_o'].to(self.device)
            results['rays_d_lr'] = rays_lr['rays_d'].to(self.device)
            results['h'] = H // scale
            results['w'] = W // scale
        
        toc = time.perf_counter()
        # np.save(f'./debug/error_{self.epoch}.npy', self.error_map.detach().cpu().numpy())
        # print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
        
        
        # np.save(f'debug/masks_{self.global_step}.npy', self.masks.cpu().numpy())
        return results

    def dataloader(self):
        size = len(self.poses ) 
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
    def save_poses(self, root):
        pose_dict = {}
        for i in range(len(self.img_names)):
            pose_dict[self.img_names[i][:-4]] = self.poses[i].numpy().tolist()
        with open(os.path.join(self.opt.workspace, 'pose_dir.json'), "w+") as f:
            json.dump(pose_dict, f, indent=4)