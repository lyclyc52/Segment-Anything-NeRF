import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX
import wandb

import numpy as np

import time

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim
except:  # old versions
    from torchmetrics.functional import ssim

import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips

from torchvision import transforms
from PIL import Image



def affinity_matrix(X):
    X_norm = F.normalize(X, dim=1)
    A = torch.mm(X_norm, X_norm.t())
    return A


def overlay_mask(image, mask, alpha=0.7, color=[1, 0, 0]):
    # image: [H, W, 3]
    # mask: [H, W]
    over_image = image.clone()
    over_image[mask] = torch.tensor(
        color, device=image.device, dtype=image.dtype)
    return image * alpha + over_image * (1 - alpha)


def overlay_mask_only(instance_id, color_map=None, render_id=-1):
    H, W = instance_id.shape
    instance_id = instance_id.reshape(H*W)

    if render_id == -1:
        color_mask = color_map[instance_id]
        color_mask = color_mask.reshape(H, W, -1)
    else:
        mask = instance_id == render_id
        color_mask = torch.zeros([H, W, 3], device=instance_id.device)
        color_mask[mask] = torch.tensor(color_map[render_id])
    return color_mask


def overlay_mask_composition(image, instance_id, color_map=None, render_id=-1, alpha=0.7):
    H, W = instance_id.shape
    instance_id_flatten = instance_id.reshape(H*W)
    color_mask = color_map[instance_id_flatten]
    color_mask = color_mask.reshape(H, W, -1)
    if render_id != -1:
        mask = instance_id != render_id
        color_mask[mask] = image[mask]
    return image * alpha + color_mask * (1 - alpha)


def overlay_mask_heatmap(mask, instance_id, color_map=None, alpha=0.7):
    # image: [H, W, 3]
    # mask: [H, W]

    if isinstance(instance_id, int):
        instance_id = torch.ones_like(mask) * instance_id
        instance_id = instance_id.to(color_map.device).to(torch.long)
    H, W = instance_id.shape
    instance_id = instance_id.reshape(H*W)
    color_mask = color_map[instance_id]
    color_mask = color_mask.reshape(H, W, -1)

    output = color_mask * mask[..., None]

    return output


def overlay_point(image, points, radius=2, color=[0, 1, 0]):
    # image: [H, W, 3]
    # points: [1, 2]
    mask = torch.zeros_like(image[:, :, 0]).bool()
    for point in points:
        mask[point[1]-radius:point[1]+radius,
             point[0]-radius:point[0]+radius] = True
    image[mask] = torch.tensor(color, device=image.device, dtype=image.dtype)
    return image


def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0]
                                                                 and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(
                y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]


def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')




@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None, device='cpu', incoherent_mask=None, 
             include_incoherent_region=False, incoherent_mask_size = 128, random_sample=False):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''

    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:,
                                                      1], intrinsics[:, 2], intrinsics[:, 3]

    i, j = custom_meshgrid(torch.linspace(
        0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))  # float
    
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5


    results = {}
    

    if N > 0:
        
        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]

        elif patch_size > 1 and not random_sample:
            if incoherent_mask is not None and include_incoherent_region:
                
                inds_coarse_center = torch.multinomial(incoherent_mask.to(torch.float32), 1)
                
                inds_x, inds_y = inds_coarse_center // incoherent_mask_size, inds_coarse_center % incoherent_mask_size
                
                
                # rand_indices = torch.multinomial(incoherent_mask.view(-1, H*W).to(torch.float32), 1)
                sx, sy = H / incoherent_mask_size, W / incoherent_mask_size
                mask_point_x = inds_x * sx
                mask_point_y = inds_y * sy

                # print(mask_point_x, mask_point_y)
                inds_x = torch.clamp(mask_point_x-patch_size//2, min=0, max=H-patch_size-1).long()
                inds_y = torch.clamp(mask_point_y-patch_size//2, min=0, max=W-patch_size-1).long()
                
            
            


                
            else:
                # random sample left-top cores.
                
                num_patch = N // (patch_size ** 2)
                inds_x = torch.randint(
                    0, H - patch_size, size=[num_patch], device=device)
                inds_y = torch.randint(
                    0, W - patch_size, size=[num_patch], device=device)
            
            

            inds = torch.stack([inds_x, inds_y], dim=-1)  # [num_sample, 1, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(
                patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack(
                [pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            # inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  
            inds = inds + offsets.unsqueeze(0)  # [num_sample, p^2, 2]


            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[..., 0] * W + inds[..., 1]  # [N], flatten
            
            


        else:  # random sampling
            inds = torch.randint(
                0, H*W, size=[N], device=device)  # may duplicate

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['i'] = i.long()
        results['j'] = j.long()
        
    else:
        inds = torch.arange(H*W, device=device)

    zs = -torch.ones_like(i)  # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy  # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1)  # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]
    

    rays_d = (directions.unsqueeze(1) @
              poses[:, :3, :3].transpose(-1, -2)).squeeze(1)

    rays_o = poses[:, :3, 3].expand_as(rays_d)  # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    
    
    
    

    if incoherent_mask is not None and include_incoherent_region:
        inds_x, inds_y = inds // W, inds % W
        sx_coarse, sy_coarse = incoherent_mask_size / H, incoherent_mask_size / W
        inds_coarse_x = (inds_x * sx_coarse).long()
        inds_coarse_y = (inds_y * sy_coarse).long()
    
        results['inds_coarse'] = (inds_coarse_x * incoherent_mask_size + inds_coarse_y).long()

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results



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


def visualize_rays(rays_o, rays_d):

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(0, rays_o.shape[0], 10):
        ro = rays_o[i]
        rd = rays_d[i]

        segs = np.array([[ro, ro + rd * 3]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        # [B, N, 3] or [B, H, W, 3], range[0, 1]
        preds, truths = self.prepare_inputs(preds, truths)

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

        return psnr

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"),
                          self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='vgg', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # [H, W, 3] --> [B, 3, H, W], range in [0, 1]
        preds, truths = self.prepare_inputs(preds, truths)
        # normalize=True: [0, 1] to [-1, 1]
        v = self.fn(truths, preds, normalize=True).item()
        self.V += v
        self.N += 1

        return v

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(
            prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        preds, truths = self.prepare_inputs(preds, truths)

        v = ssim(preds, truths)

        self.V += v
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"),
                          self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class MeanIoUMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy().astype(np.int64)
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N] or [B, H, W], range: [0, num_classes-1]
        num_classes = max(preds.max() + 1, truths.max() + 1)
          
        ious = []
        for i in range(num_classes):
            intersection = np.logical_and(preds == i, truths == i).sum()
            union = np.logical_or(preds == i, truths == i).sum()

            if union > 0:
                ious.append(intersection / union)
        
        self.V += np.mean(ious)
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "mIoU"), self.measure(), global_step)

    def report(self):
        return f'mIoU = {self.measure():.6f}'
    
    def name(self):
        return 'mIoU'


class Cache:
    def __init__(self, size=100):
        self.size = size
        self.data = {}
        self.key = 0

    def full(self):
        return len(self.data) == self.size

    def insert(self, x):
        self.data[self.key] = x
        self.key = (self.key + 1) % self.size

    def get(self, key=None):
        if key is None:
            key = random.randint(0, len(self.data) - 1)
        return self.data[key]


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 metrics=[],
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 # device to use, usually setting to None is OK. (auto choose device)
                 device=None,
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 # save once every $ epoch (independently from eval)
                 save_interval=1,
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 # whether to call scheduler.step() after every train step
                 scheduler_update_every_step=False,
                 sam_predictor=None,
                 ):

        self.opt = opt
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.sam_predictor = sam_predictor
        # for GUI
        self.point_3d = None
        self.last_masks = None
        # for cache
        self.cache = Cache(self.opt.cache_size)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.color_map = np.multiply([
            plt.cm.get_cmap('gist_ncar', 41)((i * 7 + 5) % 41)[:3] for i in range(41)
        ], 1)
        self.color_map = torch.from_numpy(
            self.color_map).to(self.device).to(torch.float)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(
            f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        self.log(opt)
        self.log(self.model)

        if self.workspace is not None:

            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(
                        f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    # ------------------------------

    # rgb loss version 0
    # def rgb_similarity_loss(self, rgb, inst_masks, gt_masks, incoherent_masks = None):
        
    #     pred_masks = torch.argmax(inst_masks, -1)
    #     # random sample some points
    #     if incoherent_masks is None:
    #         perm = torch.randperm(rgb.size(0))
    #         sample_index = perm[:self.opt.rgb_similarity_num_sample]
    #     else:
            
    #         # sample_index_weight = (1. - torch.logical_xor(pred_masks, gt_masks).to(torch.float32)) * (1.-incoherent_masks.to(torch.float32))
            
    #         sample_index_weight = torch.logical_and(pred_masks, gt_masks).to(torch.float32)
            
    #         if sample_index_weight.sum() == 0:
    #             # sample_index_weight = (1. - incoherent_masks).to(torch.float32) 
    #             sample_index_weight = torch.ones_like(sample_index_weight)          
    #         num_sample = self.opt.rgb_similarity_num_sample if self.opt.rgb_similarity_num_sample < sample_index_weight.sum() else sample_index_weight.sum().to(torch.int32)
            
    #         # if (1. - torch.logical_xor(pred_masks, gt_masks).to(torch.float32)).sum() < 0
    #         sample_index = torch.multinomial(sample_index_weight, num_sample, replacement=False)
        
        
    #     rgb_sample = rgb[sample_index][:, None, ...]
        
    #     sample_mask = inst_masks[sample_index][:, None, ...].detach()
    #     sample_mask_arg =  torch.argmax(sample_mask, -1)
    #     sample_mask = torch.zeros_like(sample_mask, device=sample_mask.device)
    #     sample_mask= sample_mask.scatter_(-1, sample_mask_arg[..., None], 1)
    #     sample_mask_args = pred_masks[sample_index]

    #     # gt_masks
        
    #     rgb = rgb[None, ...]
    #     inst_masks = inst_masks[None, ...]
        
    #     color_similarity_map = torch.norm(rgb-rgb_sample, dim=-1)
    #     similarity_map = color_similarity_map < self.opt.rgb_similarity_threshold
        

        
    #     if self.opt.redundant_instance > 0:
    #         pred_masks_similarity = F.cosine_similarity(inst_masks, sample_mask, dim=-1)
    #         pred_masks_similarity = torch.exp(- self.opt.rgb_similarity_exp_weight * pred_masks_similarity - self.opt.epsilon) 
    #         labels = 1. - similarity_map.to(torch.int)
    #         rgb_loss = F.binary_cross_entropy(pred_masks_similarity, labels, reduction='mean')
    #     else:
    #         # rgb_similar_mask = inst_masks[similarity_map]
    #         pred_masks_similarity = F.cosine_similarity(inst_masks, sample_mask, dim=-1)
    #         pred_masks_similarity = torch.exp(- self.opt.rgb_similarity_exp_weight * pred_masks_similarity - self.opt.epsilon) 
    #         pred_masks_similarity = (similarity_map * pred_masks_similarity).sum(-1)
    #         mean_weight = similarity_map.sum(-1)
            
    #         # print(pred_masks_similarity.shape)
    #         # print(similarity_map.shape)
    #         rgb_loss = (pred_masks_similarity / mean_weight).mean()
            

    #     # label = torch.zeros(pred_masks_similarity.shape[0], requires_grad=False).to(pred_masks_similarity.device)
    #     # loss = F.binary_cross_entropy(pred_masks_similarity, label, reduction='mean')

        
    #     return rgb_loss
    
    # rgb loss version 2
    def rgb_similarity_loss(self, rgb, inst_masks, gt_flattened, use_pred_logistics = False):
        '''
        Args:
            local_*** : [num of local samples, local patch size ^2, -1]
            use_pred_logistics
        '''

        # random sample some points
        
        weights = torch.ones(rgb.shape[1]).expand(rgb.shape[0], -1).to(rgb.device)
        sample_index = torch.multinomial(weights, num_samples=self.opt.rgb_similarity_num_sample, replacement=False)
        
        
        col_ids = torch.arange(rgb.shape[0], dtype=torch.int64).to(rgb.device)
        rgb_sample = rgb[col_ids[:, None], sample_index][..., None, :]
        

        
        sample_mask = inst_masks[col_ids[:, None], sample_index][..., None, :].detach()
        sample_mask_gt = gt_flattened[col_ids[:, None], sample_index].detach()
        if not use_pred_logistics:
            sample_mask_arg =  torch.argmax(sample_mask, -1)
            sample_mask = torch.zeros_like(sample_mask, device=sample_mask.device)
            sample_mask= sample_mask.scatter_(-1, sample_mask_gt[..., None], 1)


        # gt_masks
        
        rgb = rgb[:, None, ...]
        inst_masks = inst_masks[:, None, ...]


        # calculate indices that has similar rgb values
        color_similarity_map = torch.norm(rgb-rgb_sample, dim=-1)
        similarity_map = color_similarity_map < self.opt.rgb_similarity_threshold


        
        if self.opt.redundant_instance > 0:
            pred_masks_similarity = F.cosine_similarity(inst_masks, sample_mask, dim=-1)
            pred_masks_similarity = torch.exp(- self.opt.rgb_similarity_exp_weight * pred_masks_similarity - self.opt.epsilon) 
            labels = 1. - similarity_map.to(torch.int)
            rgb_loss = F.binary_cross_entropy(pred_masks_similarity, labels, reduction='mean')
        else:
            # rgb_similar_mask = inst_masks[similarity_map]
            pred_masks_similarity = F.cosine_similarity(inst_masks, sample_mask, dim=-1)
            pred_masks_similarity = torch.exp(- self.opt.rgb_similarity_exp_weight * pred_masks_similarity - self.opt.epsilon) 
            
            pred_masks_similarity = (similarity_map * pred_masks_similarity).sum(-1)
            

            mean_weight = similarity_map.sum(-1)


            rgb_loss = (pred_masks_similarity / mean_weight).mean()

            
        # label = torch.zeros(pred_masks_similarity.shape[0], requires_grad=False).to(pred_masks_similarity.device)
        # loss = F.binary_cross_entropy(pred_masks_similarity, label, reduction='mean')

        
        return rgb_loss
    
    def label_regularization(self, depth, pred_masks):
        '''
        depth: [B, N]
        pred_masks: [B, N, num_instances]
        '''
        pred_masks = pred_masks.view(-1, self.opt.patch_size, self.opt.patch_size,
                                     self.opt.n_inst).permute(0, 3, 1, 2).contiguous()  # [B, num_instances, patch_size, patch_size]

        diff_x = pred_masks[:, :, :, 1:] - pred_masks[:, :, :, :-1]
        diff_y = pred_masks[:, :, 1:, :] - pred_masks[:, :, :-1, :]

        # [B, patch_size, patch_size]
        depth = depth.view(-1, self.opt.patch_size, self.opt.patch_size)

        depth_diff_x = depth[:, :, 1:] - depth[:, :, :-1]
        depth_diff_y = depth[:, 1:, :] - depth[:, :-1, :]
        weight_x = torch.exp(-(depth_diff_x * depth_diff_x)
                             ).unsqueeze(1).expand_as(diff_x)
        weight_y = torch.exp(-(depth_diff_y * depth_diff_y)
                             ).unsqueeze(1).expand_as(diff_y)

        diff_x = diff_x * diff_x * weight_x
        diff_y = diff_y * diff_y * weight_y

        smoothness_loss = torch.sum(
            diff_x) / torch.sum(weight_x) + torch.sum(diff_y) / torch.sum(weight_y)

        return smoothness_loss

    def train_step(self, data):

        # use cache instead of novel poses
        use_cache = self.opt.with_sam and \
            self.opt.cache_size > 0 and \
            self.cache.full() and \
            self.global_step % self.opt.cache_interval != 0

        # override data
        if use_cache:
            data = self.cache.get()

        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None
        H, W = data['H'], data['W']

        N = rays_o.shape[0]
        if self.opt.background == 'random':
            # [N, 3], pixel-wise random.
            bg_color = torch.rand(N, 3, device=self.device)
        else:  # white / last_sample
            bg_color = 1

        # rgb training
        if not self.opt.with_sam and not self.opt.with_mask:

            images = data['images']  # [N, 3/4]

            C = images.shape[-1]
            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + \
                    bg_color * (1 - images[..., 3:])
            else:
                gt_rgb = images

            update_proposal = (not self.opt.with_sam) and (
                self.global_step <= 3000 or self.global_step % 5 == 0)

            outputs = self.model.render(rays_o, rays_d, staged=False, index=index, bg_color=bg_color,
                                        perturb=True, cam_near_far=cam_near_far, update_proposal=update_proposal, return_feats=0)
            pred_rgb = outputs['image']

            loss = self.criterion(pred_rgb, gt_rgb).mean()

            # extra loss
            if 'proposal_loss' in outputs and self.opt.lambda_proposal > 0:
                loss = loss + self.opt.lambda_proposal * \
                    outputs['proposal_loss']

            if 'distort_loss' in outputs and self.opt.lambda_distort > 0:
                loss = loss + self.opt.lambda_distort * outputs['distort_loss']

            if self.opt.lambda_entropy > 0:
                w = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
                entropy = - w * torch.log2(w) - (1 - w) * torch.log2(1 - w)
                loss = loss + self.opt.lambda_entropy * (entropy.mean())

            # adaptive num_rays
            if self.opt.adaptive_num_rays:
                self.opt.num_rays = int(
                    round((self.opt.num_points / outputs['num_points']) * self.opt.num_rays))

            return pred_rgb, gt_rgb, loss

        # online distillation of SAM features
        else:
            if self.opt.with_mask:
                gt_mask = data['masks'].to(torch.long)  # [B, N], N=H*W?
                B, N = gt_mask.shape
                # num_instances = torch.unique(gt_masks).shape[0]

                bg_color = 1

                outputs = self.model.render(rays_o, rays_d, staged=False, index=index, bg_color=bg_color,
                                            perturb=False, cam_near_far=cam_near_far, update_proposal=False, 
                                            return_rgb=0, return_feats=0, return_mask=1)

                # [B, N, num_instances]
                inst_masks = outputs['instance_mask_logits']
 
                # [B*N, num_instances]
                
                gt_masks_flattened = gt_mask.view(-1)  # [B*N]

                labeled = gt_masks_flattened != -1  # only compute loss for labeled pixels
                
                inst_masks = torch.softmax(
                    inst_masks, dim=-1)  # [B, N, num_instances + k]
                pred_masks = torch.stack([inst_masks[..., :-1].sum(-1), inst_masks[..., -1]], -1)
                
                pred_masks_flattened = pred_masks.view(-1, 2)
                pred_masks_flattened = torch.clamp(pred_masks_flattened, min=self.opt.epsilon, max=1-self.opt.epsilon)
 
                if labeled.sum() > 0:
                    # [B*N], loss fn with reduction='none'
                    loss = -torch.log(torch.gather(pred_masks_flattened[labeled], -1, gt_masks_flattened[labeled][..., None]))
                    
                else:
                    loss = torch.tensor(0).to(
                        pred_masks_flattened.dtype).to(self.device)
                
                if self.opt.incoherent_uncertainty_weight < 1:
                    loss = (1 - data['incoherent_masks'] + self.opt.incoherent_uncertainty_weight * data['incoherent_masks']) * loss
                
                if self.error_map is not None:
                    index = data['index'] # [B]
                    inds = data['inds_coarse']# [B]
                    

                    # take out, this is an advanced indexing and the copy is unavoidable.
                    


                    
                    sample_mask_gt = torch.zeros_like(pred_masks_flattened, device=pred_masks_flattened.device)
                    sample_mask_gt= sample_mask_gt.scatter_(-1, gt_masks_flattened[..., None], 1)
                    
                    pred_masks_similarity = F.cosine_similarity(inst_masks, sample_mask_gt, dim=-1)
                    error = torch.exp(- self.opt.rgb_similarity_exp_weight * pred_masks_similarity - self.opt.epsilon) 
                    # ema update
                    # print(error_map.shape)
                    # print(inds.shape)
                    
                    ema_error = 0.1 * self.error_map[index, inds] + 0.9 * error
                    self.error_map[index, inds] = ema_error
                    
                    
                    # np.save(f'./debug/error_{self.epoch}.npy', self.train_loader._data.error_map.detach().cpu().numpy())
                    # put back
                    # self.error_map[index] = error_map
                loss = loss.mean()
                    
                if self.opt.label_regularization_weight > 0:
                    loss = loss + self.label_regularization(
                        outputs['depth'].detach(), pred_masks) * self.opt.label_regularization_weight
                    
                    
                if self.opt.rgb_similarity_loss_weight > 0 and self.global_step > self.opt.rgb_similarity_iter:
                    if self.opt.mixed_sampling:
                        
                        local_inst_masks = inst_masks[self.opt.num_rays:]
                        local_inst_masks = local_inst_masks.view(self.opt.num_local_sample, self.opt.local_sample_patch_size*self.opt.local_sample_patch_size, -1)
                        
                        local_rgb = outputs['image'][self.opt.num_rays:]
                        local_rgb = local_rgb.view(self.opt.num_local_sample, self.opt.local_sample_patch_size*self.opt.local_sample_patch_size, -1)
                        local_gt_flattened = gt_masks_flattened[self.opt.num_rays:]
                        local_gt_flattened = local_gt_flattened.view(self.opt.num_local_sample, self.opt.local_sample_patch_size*self.opt.local_sample_patch_size, -1)
                        

                        loss = loss + self.rgb_similarity_loss(local_rgb, local_inst_masks, local_gt_flattened, use_pred_logistics = self.opt.rgb_similarity_use_pred_logistics) \
                                                                   * self.opt.rgb_similarity_loss_weight
                       
                        


                    else:
                        loss = loss + self.rgb_similarity_loss(outputs['image'].detach()[None, ...], inst_masks[None, ...], 
                                                               gt_masks_flattened[None, ...], use_pred_logistics = self.opt.rgb_similarity_use_pred_logistics) \
                                                                   * self.opt.rgb_similarity_loss_weight
                       
                    
                pred_masks = pred_masks.argmax(dim=-1)  # [B, N]
                    

                return pred_masks, gt_mask, loss

            elif self.opt.with_sam:
                with torch.no_grad():
                    if use_cache:
                        gt_samvit = data['gt_samvit']
                    else:
                        # render high-res RGB
                        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color,
                                                    perturb=True, cam_near_far=cam_near_far, update_proposal=False, return_feats=0)
                        pred_rgb = outputs['image'].reshape(H, W, 3)

                        # encode SAM ground truth
                        image = (pred_rgb.detach().cpu().numpy()
                                 * 255).astype(np.uint8)
                        self.sam_predictor.set_image(image)
                        # [1, 256, 64, 64]
                        gt_samvit = self.sam_predictor.features

                        # write to cache
                        if self.opt.cache_size > 0:
                            data['gt_samvit'] = gt_samvit
                            self.cache.insert(data)

                # always use 64x64 features as SAM default to 1024x1024
                h, w = data['h'], data['w']
                rays_o_hw = data['rays_o_lr']
                rays_d_hw = data['rays_d_lr']
                outputs = self.model.render(rays_o_hw, rays_d_hw, staged=False, index=index, bg_color=bg_color,
                                            perturb=False, cam_near_far=cam_near_far, return_feats=1, H=h, W=w)
                pred_samvit = outputs['samvit'].reshape(
                    1, h, w, 256).permute(0, 3, 1, 2).contiguous()
                pred_samvit = F.interpolate(
                    pred_samvit, gt_samvit.shape[2:], mode='bilinear')

                # loss
                loss = self.criterion(pred_samvit, gt_samvit).mean()

                return pred_samvit, gt_samvit, loss

    def post_train_step(self):
        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # the new inplace TV loss
        if self.opt.lambda_tv > 0:
            self.model.apply_total_variation(self.opt.lambda_tv)

        if self.opt.lambda_wd > 0:
            self.model.apply_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None
        H, W = data['H'], data['W']
        bg_color = 1

        # full resolution RGBD query, do not query feats!
        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color, perturb=False,
                                    cam_near_far=cam_near_far, return_feats=0, return_mask=self.opt.with_mask)

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)

        if not self.opt.with_sam and not self.opt.with_mask:
            images = data['images']  # [H, W, 3/4]
            C = images.shape[-1]
            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + \
                    bg_color * (1 - images[..., 3:])
            else:
                gt_rgb = images

            loss = self.criterion(pred_rgb, gt_rgb).mean()

            return pred_rgb, pred_depth, None, gt_rgb, loss

        else:
            if self.opt.with_mask:
                gt_mask = data['masks'].to(torch.long)
      
                # [B*H*W, num_instances]
                # pred_mask_flattened = pred_mask.view(-1, self.opt.n_inst)
                gt_mask_flattened = gt_mask.view(-1)  # [B*H*W]

                labeled = gt_mask_flattened != -1  # only compute loss for labeled pixels
                
                inst_mask = outputs['instance_mask_logits'].reshape(
                H, W, self.opt.n_inst + self.opt.redundant_instance)
            
                if self.opt.n_inst > 1:
                    inst_mask = torch.softmax(inst_mask, dim=-1)                
                    pred_mask = torch.stack([inst_mask[..., :-1].sum(-1), inst_mask[..., -1]], -1)
                else:
                    pred_mask = torch.sigmoid(inst_mask)

                pred_mask_flattened = pred_mask.view(-1, 2)
 
                pred_mask_flattened = torch.clamp(pred_mask_flattened, min=self.opt.epsilon, max=1-self.opt.epsilon)
 
                if labeled.sum() > 0:
                    loss = -torch.log(torch.gather(pred_mask_flattened[labeled], -1, gt_mask_flattened[labeled][..., None]))
                else:
                    loss = torch.tensor(0).to(
                        pred_mask_flattened.dtype).to(self.device)
                    
                loss = loss.mean()

                if self.opt.label_regularization_weight > 0:
                    loss = loss + self.label_regularization(
                        outputs['depth'].detach(), pred_mask) * self.opt.label_regularization_weight

                # [B, H, W, num_instances]
                # if self.opt.n_inst > 1: 
                #     pred_mask = torch.softmax(pred_mask, dim=-1)
                # else:
                #     pred_masks = torch.sigmoid(pred_masks)
                # pred_mask = pred_mask.argmax(dim=-1) # [B, H, W]

                # pred_seg = overlay_mask(pred_rgb, pred_mask)

                # gt_seg = overlay_mask(pred_rgb, gt_mask)

                return pred_rgb, pred_depth, pred_mask, gt_mask, loss

            if self.opt.with_sam:
                # encode SAM ground truth
                image = (pred_rgb.detach().cpu().numpy()
                         * 255).astype(np.uint8)
                self.sam_predictor.set_image(image)
                gt_samvit = self.sam_predictor.features

                # always use 64x64 features as SAM default to 1024x1024
                h, w = data['h'], data['w']
                rays_o_hw = data['rays_o_lr']
                rays_d_hw = data['rays_d_lr']
                outputs = self.model.render(rays_o_hw, rays_d_hw, staged=False, index=index, bg_color=bg_color,
                                            perturb=False, cam_near_far=cam_near_far, return_feats=1, H=h, W=w)
                pred_samvit = outputs['samvit'].reshape(
                    1, h, w, 256).permute(0, 3, 1, 2).contiguous()
                pred_samvit = F.interpolate(
                    pred_samvit, gt_samvit.shape[2:], mode='bilinear')

                # report feature loss
                loss = self.criterion(pred_samvit, gt_samvit).mean()

                # TODO: grid point samples to evaluate IoU...
                if self.opt.use_point:
                    masks, point_coords, low_res_masks = self.sam_predict(
                        H, W, pred_samvit)

                    pred_seg = overlay_mask(pred_rgb, masks[0])
                    pred_seg = overlay_point(pred_seg, point_coords)

                    # gt_masks, point_coords, low_res_masks = self.sam_predict(H, W, gt_samvit, point_coords, image=(pred_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
                    gt_masks, point_coords, low_res_masks = self.sam_predict(
                        H, W, gt_samvit, point_coords)  # use gt feature to debug

                    gt_seg = overlay_mask(pred_rgb, gt_masks[0])
                    gt_seg = overlay_point(gt_seg, point_coords)

                    return pred_seg, pred_depth, pred_samvit, gt_seg, loss
                else:
                    return pred_rgb, pred_depth, pred_samvit, pred_rgb, loss

    def test_step(self, data, bg_color=None, perturb=False, point_coords=None):
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        H, W = data['H'], data['W']

        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        # full resolution RGBD query, do not query feats!
        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color,
                                    perturb=perturb, cam_near_far=cam_near_far, return_feats=False, return_mask=self.opt.with_mask)

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)

        if self.opt.with_mask:
            inst_mask = outputs['instance_mask_logits'].reshape(
                H, W, self.opt.n_inst + self.opt.redundant_instance)
            
            if self.opt.n_inst > 1:
                inst_mask = torch.softmax(inst_mask, dim=-1)                
                pred_mask = torch.stack([inst_mask[..., :-1].sum(-1), inst_mask[..., -1]], -1)
            else:
                pred_mask = torch.sigmoid(inst_mask)

            if self.opt.render_mask_type == 'heatmap':
                if self.opt.render_mask_instance_id >= 0 and self.opt.render_mask_instance_id < self.opt.n_inst:
                    instance_mask = pred_mask[...,
                                              self.opt.render_mask_instance_id]
                    instance_id = self.opt.render_mask_instance_id
                else:
                    instance_mask, _ = torch.max(pred_mask, -1)
                    instance_id = pred_mask.argmax(-1)

                pred_rgb = overlay_mask_heatmap(
                    instance_mask, instance_id, color_map=self.color_map)
            elif self.opt.render_mask_type == 'composition':
                instance_mask, _ = torch.max(pred_mask, -1)
                instance_id = pred_mask.argmax(-1)

                if self.opt.render_mask_instance_id >= 0 and self.opt.render_mask_instance_id < self.opt.n_inst:
                    render_id = self.opt.render_mask_instance_id
                else:
                    render_id = -1

                pred_rgb = overlay_mask_composition(
                    pred_rgb, instance_id, color_map=self.color_map, render_id=render_id)
            elif self.opt.render_mask_type == 'mask':
                instance_mask, _ = torch.max(pred_mask, -1)
                instance_id = pred_mask.argmax(-1)

                if self.opt.render_mask_instance_id >= 0 and self.opt.render_mask_instance_id < self.opt.n_inst:
                    render_id = self.opt.render_mask_instance_id
                else:
                    render_id = -1

                pred_rgb = overlay_mask_only(
                    instance_id, color_map=self.color_map, render_id=render_id)

        if self.opt.with_sam:
            h, w = data['h'], data['w']
            rays_o_hw = data['rays_o_lr']
            rays_d_hw = data['rays_d_lr']
            outputs = self.model.render(rays_o_hw, rays_d_hw, staged=False, index=index, bg_color=bg_color,
                                        perturb=False, cam_near_far=cam_near_far, return_feats=1, H=h, W=w)
            pred_samvit = outputs['samvit'].reshape(
                1, h, w, 256).permute(0, 3, 1, 2).contiguous()

            # remember new point_3d
            if point_coords is not None:
                rays_o = rays_o.view(H, W, 3)
                rays_d = rays_d.view(H, W, 3)
                point_depth = pred_depth[point_coords[:,
                                                      1], point_coords[:, 0]]
                point_rays_o = rays_o[point_coords[:, 1], point_coords[:, 0]]
                point_rays_d = rays_d[point_coords[:, 1], point_coords[:, 0]]
                point_3d = point_rays_o + point_rays_d * \
                    point_depth.unsqueeze(-1)  # [1, 3]

                # update current selected points
                if self.point_3d is None:
                    self.point_3d = point_3d
                else:
                    dist = (self.point_3d - point_3d).norm(dim=-1)
                    dist_thresh = 0.01
                    if dist.min() > dist_thresh:
                        # add if not close to any existed point
                        # print(f'[INFO] add new point {point_3d}')
                        self.point_3d = torch.cat(
                            [self.point_3d, point_3d], dim=0)
                    else:
                        # remove existed point if too close
                        # print(f'[INFO] remove old point mask {dist <= dist_thresh}')
                        keep_mask = dist > dist_thresh
                        if keep_mask.any():
                            self.point_3d = self.point_3d[keep_mask]
                        else:
                            self.point_3d = None

            # get remembered points coords first
            inputs_point_coords = None
            if self.point_3d is not None:
                print(self.point_3d)
                print()
                point_3d = torch.cat([self.point_3d, torch.ones_like(
                    self.point_3d[:, :1])], axis=-1)  # [N, 4]
                w2c = torch.inverse(data['poses'][0])  # [4, 4]
                point_3d_cam = point_3d @ w2c.T  # [N, 4]
                intrinsics = data['intrinsics'][0]  # [4]
                fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
                inputs_point_coords = torch.stack([
                    W - (fx * point_3d_cam[:, 0] / point_3d_cam[:, 2] + cx),
                    fy * point_3d_cam[:, 1] / point_3d_cam[:, 2] + cy,
                ], dim=-1).long()  # [N, 2]
                # mask out-of-screen coords
                screen_mask = (inputs_point_coords[:, 0] >= 0) & (inputs_point_coords[:, 0] < W) & (
                    inputs_point_coords[:, 1] >= 0) & (inputs_point_coords[:, 1] < H)
                if screen_mask.any():
                    inputs_point_coords = inputs_point_coords[screen_mask]
                    # depth test to reject those occluded point_coords
                    point_depth = - point_3d_cam[screen_mask, 2]
                    observed_depth = pred_depth[inputs_point_coords[:,
                                                                    1], inputs_point_coords[:, 0]]
                    unoccluded_mask = (
                        point_depth - observed_depth).abs() <= 0.05
                    if unoccluded_mask.any():
                        inputs_point_coords = inputs_point_coords[unoccluded_mask].detach(
                        ).cpu().numpy()
                    else:
                        inputs_point_coords = None
                else:
                    inputs_point_coords = None

            if inputs_point_coords is not None:
                # masks, outputs_point_coords, low_res_masks = self.sam_predict(H, W, pred_samvit, inputs_point_coords, image=(pred_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
                masks, outputs_point_coords, low_res_masks = self.sam_predict(
                    H, W, pred_samvit, inputs_point_coords)

                pred_rgb = overlay_mask(pred_rgb, masks[0])
                pred_rgb = overlay_point(pred_rgb, outputs_point_coords)

        return pred_rgb, pred_depth

    def sam_predict(self, H, W, features, point_coords=None, mask_input=None, image=None):
        # H/W: original image size
        # features: [1, 256, h, w]
        # point_coords: [N, 2] np.ndarray, int32
        # image: np.ndarray [H, W, 3], uint8, debug use, if provided, override with GT feature

        resize_ratio = 1024 / W if W > H else 1024 / H
        input_size = (int(H * resize_ratio), int(W * resize_ratio))

        if image is not None:
            self.sam_predictor.set_image(image)
        else:
            # mimic set_image
            self.sam_predictor.reset_image()
            self.sam_predictor.original_size = (H, W)
            self.sam_predictor.input_size = input_size

            h, w = features.shape[2:]
            resize_ratio_feat = 64 / w if w > h else 64 / h
            features = F.interpolate(features, (int(
                h * resize_ratio_feat), int(w * resize_ratio_feat)), mode='bilinear', align_corners=False)
            features = F.pad(
                features, (0, 64 - features.shape[3], 0, 64 - features.shape[2]), mode='constant', value=0)
            self.sam_predictor.features = features
            self.sam_predictor.is_image_set = True

        if point_coords is None:
            # random single point if not provided
            border_h = int(input_size[0] * 0.2)
            border_w = int(input_size[1] * 0.2)
            point_coords = np.array([[
                np.random.randint(0 + border_h, input_size[1] - border_h),
                np.random.randint(0 + border_w, input_size[0] - border_w)
            ]])
        else:
            # scale to input size
            point_coords = (point_coords.astype(np.float32)
                            * resize_ratio).astype(np.int32)

        # use last mask as a prior if provided
        # NOTE: seems not useful, still need the point inputs...
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        else:
            mask_input_torch = None

        point_labels = np.ones_like(point_coords[:, 0])  # [N]
        coords_torch = torch.as_tensor(
            point_coords, dtype=torch.float, device=self.device)
        labels_torch = torch.as_tensor(
            point_labels, dtype=torch.int, device=self.device)
        coords_torch, labels_torch = coords_torch[None,
                                                  :, :], labels_torch[None, :]

        # decode
        masks, iou_predictions, low_res_masks = self.sam_predictor.predict_torch(
            coords_torch, labels_torch,
            mask_input=mask_input_torch,
            multimask_output=False,
        )

        original_point_coords = (point_coords / resize_ratio).astype(np.int32)
        # [N, H, W], [N, 2], [N, 256, 256]
        return masks[0], original_point_coords, low_res_masks[0]

    # ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name))

        start_t = time.time()


        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            train_loader.epoch = epoch
            self.train_one_epoch(train_loader)
            self.save_interval = 1
            if (self.epoch % self.save_interval == 0 or self.epoch == max_epochs) and self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False, remove_old=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.6f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    preds, preds_depth = self.test_step(data)
                pred = preds.detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth.detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / \
                    (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(
                        pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(
                        save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)  # [N, H, W, 3]
            all_preds_depth = np.stack(all_preds_depth, axis=0)  # [N, H, W]

            # fix ffmpeg not divisible by 2
            all_preds = np.pad(all_preds, ((0, 0), (0, 1 if all_preds.shape[1] % 2 != 0 else 0), (
                0, 1 if all_preds.shape[2] % 2 != 0 else 0), (0, 0)))
            all_preds_depth = np.pad(all_preds_depth, ((
                0, 0), (0, 1 if all_preds_depth.shape[1] % 2 != 0 else 0), (0, 1 if all_preds_depth.shape[2] % 2 != 0 else 0)))

            imageio.mimwrite(os.path.join(
                save_path, f'{name}_rgb.mp4'), all_preds, fps=24, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(
                save_path, f'{name}_depth.mp4'), all_preds_depth, fps=24, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                preds, truths, loss_net = self.train_step(data)

            loss = loss_net
            if self.opt.use_wandb:
                wandb.log({"loss": loss})
            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss_net.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    # [GUI] test on a single image

    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, user_inputs=None):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1, device=self.device)

        scale = 16 * rH // 1024 if rH > rW else 16 * rW // 1024
        rays_lr = get_rays(pose, intrinsics / scale, rH //
                           scale, rW // scale, -1, device=self.device)

        data = {
            'poses': pose,
            'intrinsics': intrinsics,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'rays_o_lr': rays_lr['rays_o'],
            'rays_d_lr': rays_lr['rays_d'],
            'H': rH,
            'W': rW,
            'h': rH // scale,
            'w': rW // scale,
            'index': [0],
        }

        if user_inputs is not None:
            point_coords = user_inputs['point_coords']
        else:
            point_coords = None

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            # here spp is used as perturb random seed! (but not perturb the first sample)
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                preds, preds_depth = self.test_step(
                    data, bg_color=bg_color, perturb=False if spp == 1 else spp, point_coords=point_coords)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            preds = F.interpolate(preds.unsqueeze(0).permute(0, 3, 1, 2), size=(
                H, W), mode='nearest').permute(0, 2, 3, 1).squeeze(0).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(0).unsqueeze(
                1), size=(H, W), mode='nearest').squeeze(0).squeeze(1)

        pred = preds.detach().cpu().numpy()
        pred_depth = preds_depth.detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs



    def render_mask(self, data):
        rays_o = data['rays_o']  # [N, 3]
        rays_d = data['rays_d']  # [N, 3]
        index = data['index']  # [1/N]
        # [1/N, 2] or None
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None
        H, W = data['H'], data['W']
        bg_color = 1

        # full resolution RGBD query, do not query feats!
        outputs = self.model.render(rays_o, rays_d, staged=True, index=index, bg_color=bg_color, perturb=False,
                                    cam_near_far=cam_near_far, return_feats=0, return_mask=self.opt.with_mask)

        inst_mask = outputs['instance_mask_logits'].reshape(
                H, W, self.opt.n_inst + self.opt.redundant_instance)
            
        if self.opt.n_inst > 1:
            inst_mask = torch.softmax(inst_mask, dim=-1)                
            pred_mask = torch.stack([inst_mask[..., :-1].sum(-1), inst_mask[..., -1]], -1)
        else:
            pred_mask = torch.sigmoid(inst_mask)
        return pred_mask.argmax(-1)
        
    def update_incoherent_mask(self, loader):
        
        self.model.eval()
        rendered_mask_list = []
        for index  in range(len(loader._data.poses)):
            data = loader._data.collate_mask(index)
            mask = self.render_mask(data)
            rendered_mask_list.append(mask)
        rendered_masks = torch.stack(rendered_mask_list, 0)[..., None]
        loader._data.incoherent_masks = get_incoherent_mask(rendered_masks.permute(0,3,1,2), sfact=2)  
        loader._data.incoherent_masks = loader._data.incoherent_masks.permute(0,2,3,1).to(torch.bool)[..., 0]
        np.save('debug/files.npy', loader._data.incoherent_masks.detach().cpu().numpy())
        loader._data.incoherent_masks = loader._data.incoherent_masks.view(-1, 
                                        loader._data.incoherent_mask_size*loader._data.incoherent_mask_size)
        
        self.model.train()
        exit()
        return 
    def train_one_epoch(self, loader):
        
        self.log(
            f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        loader._data.epoch = self.epoch
        loader._data.global_step = self.global_step + 1
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0


        for data in loader:
            self.local_step += 1
            self.global_step += 1
            loader._data.global_step += 1
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                preds, truths, loss_net = self.train_step(data)

            if self.global_step % self.opt.multi_res_iter == 0 and self.global_step != 0:
                self.update_incoherent_mask(loader)
            
            loss = loss_net
            if self.opt.use_wandb:
                wandb.log({"loss": loss})
            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss_net.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar(
                        "train/loss", loss_val, self.global_step)
                    self.writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(
                        f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                pbar.update(loader.batch_size)
                
            

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}, loss={average_loss:.6f}.")
        

        
    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                image_name = data['img_names'][0]

                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    preds, preds_depth, preds_extra, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(
                        self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(
                        self.device) for _ in range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    if self.opt.with_sam or self.opt.with_mask:
                        preds_extra_list = [torch.zeros_like(preds_extra).to(
                            self.device) for _ in range(self.world_size)]  # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_extra_list, preds_depth)
                        preds_extra = torch.cat(preds_extra_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(
                        self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                if self.local_rank == 0:
                    metric_vals = []
                    for metric in self.metrics:
                        if isinstance(metric, MeanIoUMeter):
                            pred_masks = preds.argmax(-1)
                            metric_val = metric.update(pred_masks, truths)
                        else:
                            metric_val = metric.update(preds, truths)
                        metric_vals.append(metric_val)

                    # save image
                    save_path = os.path.join(
                        self.workspace, 'validation', f'{name}_{image_name}_rgb.png')
                    save_path_gt = os.path.join(
                        self.workspace, 'validation', f'{name}_{image_name}_gt.png')
                    save_path_depth = os.path.join(
                        self.workspace, 'validation', f'{name}_{image_name}_depth.npy')

                    save_path_error = None
                    if self.opt.with_sam:
                        save_path_extra = os.path.join(
                            self.workspace, 'validation', f'{name}_{image_name}_sam.npy')
                    elif self.opt.with_mask:
                        save_path_extra = os.path.join(
                            self.workspace, 'validation', f'{name}_{image_name}_mask.npy')
                    else:
                        # metric_vals[0] should be the PSNR
                        save_path_error = os.path.join(
                            self.workspace, 'validation', f'{name}_{image_name}_error_{metric_vals[0]:.2f}.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds.detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth.detach().cpu().numpy()
                    # pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    # pred_depth = (pred_depth * 255).astype(np.uint8)

                    truth = truths.detach().cpu().numpy()
                    truth = (truth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(
                        pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_gt, cv2.cvtColor(
                        truth, cv2.COLOR_RGB2BGR))
                    if save_path_error is not None:
                        error = np.abs(truth.astype(np.float32) - pred.astype(np.float32)).mean(-1).astype(np.uint8)
                        cv2.imwrite(save_path_error, error)
                    np.save(save_path_depth, pred_depth)

                    if self.opt.with_sam or self.opt.with_mask:
                        pred_extra = preds_extra.detach().cpu().numpy()
                        np.save(save_path_extra, pred_extra)

                    # if self.opt.with_mask:
                    #     pred_extra = preds_extra.detach().cpu().numpy()
                    #     np.save(save_path_extra, pred_extra)

                    pbar.set_description(
                        f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                # if max mode, use -result
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)
            else:
                # if no metric, choose best by min loss
                self.stats["results"].append(average_loss)

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(
            f"++> Evaluate epoch {self.epoch} Finished, loss = {average_loss:.6f}")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = os.path.join(
                        self.ckpt_path, self.stats["checkpoints"].pop(0))
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    # if 'density_grid' in state['model']:
                    #     del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):

        if checkpoint is None:  # load latest
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))

            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log(
                    "[WARN] No checkpoint found, abort loading latest model.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(
            f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(
                    checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
