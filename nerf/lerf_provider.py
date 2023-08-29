from nerf.colmap_provider import *
import json

def read_cameras_json(json_path):

    with open(json_path) as f:
        json_dict = json.load(f)

    return json_dict['frames']
            
        
    

class LERFDataset:
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
        self.json_path = os.path.join(self.root_path, 'transforms.json')

        camdata = read_cameras_json(self.json_path)

        # read image size (assume all images are of the same shape!)
        self.H = int(round(camdata[0]['h'] / self.downscale))
        self.W = int(round(camdata[0]['w'] / self.downscale))
        

        # read image paths

        img_names = np.array([i['file_path'] for i in camdata])
        img_paths = np.array([os.path.join(self.root_path, name[2:]) for name in img_names])


        feature_folder = os.path.join(self.root_path, 'sam_features')
        feature_paths = np.array([os.path.join(feature_folder, name.replace('jpg', 'npz') ) for name in img_names])

        # only keep existing images
        exist_mask = np.array([os.path.exists(f) for f in img_paths])
        print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
        img_paths = img_paths[exist_mask]
        feature_paths = feature_paths[exist_mask]

        # read intrinsics
        intrinsics = []
        for f in camdata:
            fl_x = f['fl_x'] / self.downscale
            fl_y = f['fl_y'] / self.downscale
            cx = f['cx'] / self.downscale
            cy = f['cy'] / self.downscale

            intrinsics.append(np.array([fl_x, fl_y, cx, cy], dtype=np.float32))
        
        self.intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]

        # read poses
        poses = []
        for f in camdata:
            P = np.array(f['transform_matrix'])
            poses.append(P)
        
        self.poses = np.stack(poses, axis=0) # [N, 4, 4]

        # read sparse points
        # ptsdata = read_points3d_binary(os.path.join(self.colmap_path, "points3D.bin"))
        # ptskeys = np.array(sorted(ptsdata.keys()))
        self.pts3d = self.poses[:, :3, 3] # [M, 3]
        
        

        # center pose
        # self.poses, self.pts3d = center_poses(poses, pts3d, self.opt.enable_cam_center)
        print(f'[INFO] ColmapDataset: load poses {self.poses.shape}, points {self.pts3d.shape}')

        # rectify convention...
        # self.poses[:, :3, 1:3] *= -1
        # self.poses = self.poses[:, [1, 0, 2, 3], :]
        # self.poses[:, 2] *= -1

        # self.pts3d = self.pts3d[:, [1, 0, 2]]
        # self.pts3d[:, 2] *= -1

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
        
            self.cam_near_far = [[0.01, 8] for _ in range(self.poses.shape[0]) ] # always extract this infomation
            
            print(f'[INFO] extracting sparse depth info...')
            # map from colmap points3d dict key to dense array index


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
            val_ids = all_ids[::8]
            # val_ids = all_ids[::50]

            if self.type == 'train':
                train_ids = np.array([i for i in all_ids if i not in val_ids]).astype(np.int32)
                self.poses = self.poses[train_ids]
                self.intrinsics = self.intrinsics[train_ids]
                img_paths = img_paths[train_ids]
                feature_paths = feature_paths[train_ids]
                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[train_ids]
            elif self.type == 'val':
                self.poses = self.poses[val_ids]
                self.intrinsics = self.intrinsics[val_ids]
                img_paths = img_paths[val_ids]
                feature_paths = feature_paths[val_ids]
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
    
        # view all poses.
        if self.opt.vis_pose:
            visualize_poses(self.poses, bound=self.opt.bound, points=self.pts3d)

        self.poses = torch.from_numpy(self.poses.astype(np.float32)) # [N, 4, 4]

        if self.images is not None:
            self.images = torch.from_numpy(self.images.astype(np.uint8)) # [N, H, W, C]
      
        if self.preload:
            self.intrinsics = self.intrinsics.to(self.device)
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                self.images = self.images.to(self.device)
            if self.cam_near_far is not None:
                self.cam_near_far = self.cam_near_far.to(self.device)

    def collate(self, index):
    
        num_rays = -1 # defaul, eval, test, train SAM use all rays

        # train RGB with random rays
        if self.training and not self.opt.with_sam:
            num_rays = self.opt.num_rays
            if self.opt.random_image_batch:
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
                fs = np.random.choice(len(self.poses), 2, replace=False)
                pose0 = self.poses[fs[0]].detach().cpu().numpy()
                pose1 = self.poses[fs[1]].detach().cpu().numpy()
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)    
                ratio = random.random()
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                poses = torch.from_numpy(pose).unsqueeze(0).to(self.device)
            # still use fixed pose, but change intrinsics
            else:
                H = W = self.opt.online_resolution
                fovy = 60
                focal = H / (2 * np.tan(0.5 * fovy * np.pi / 180))
                intrinsics = np.array([focal, focal, H / 2, W / 2], dtype=np.float32)
                intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)
    
        results = {'H': H, 'W': W}

        rays = get_rays(poses, intrinsics, H, W, num_rays, device=self.device if self.preload else 'cpu')

        if self.images is not None:
            
            if num_rays != -1:
                images = self.images[index, rays['j'], rays['i']].float() / 255 # [N, 3/4]
            else:
                images = self.images[index].squeeze(0).float() / 255 # [H, W, 3/4]

            if self.training:
                C = self.images.shape[-1]
                images = images.view(-1, C)

            results['images'] = images.to(self.device)
        
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