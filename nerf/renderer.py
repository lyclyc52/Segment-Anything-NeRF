import os
import cv2
import math
import json
import tqdm
import mcubes
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_efficient_distloss import eff_distloss


@torch.cuda.amp.autocast(enabled=False)
def distort_loss(bins, weights):
    # bins: [N, T+1]
    # weights: [N, T]

    intervals = bins[..., 1:] - bins[..., :-1]
    mid_points = bins[..., :-1] + intervals / 2

    loss = eff_distloss(weights, mid_points, intervals)

    return loss


@torch.cuda.amp.autocast(enabled=False)
def proposal_loss(all_bins, all_weights):
    # all_bins: list of [N, T+1]
    # all_weights: list of [N, T]

    def loss_interlevel(t0, w0, t1, w1):
        # t0, t1: [N, T+1]
        # w0, w1: [N, T]
        cw1 = torch.cat([torch.zeros_like(w1[..., :1]),
                        torch.cumsum(w1, dim=-1)], dim=-1)
        inds_lo = (torch.searchsorted(
            t1[..., :-1].contiguous(), t0[..., :-1].contiguous(), right=True) - 1).clamp(0, w1.shape[-1] - 1)
        inds_hi = torch.searchsorted(t1[..., 1:].contiguous(
        ), t0[..., 1:].contiguous(), right=True).clamp(0, w1.shape[-1] - 1)

        cw1_lo = torch.take_along_dim(cw1[..., :-1], inds_lo, dim=-1)
        cw1_hi = torch.take_along_dim(cw1[..., 1:], inds_hi, dim=-1)
        w = cw1_hi - cw1_lo

        return (w0 - w).clamp(min=0) ** 2 / (w0 + 1e-8)

    bins_ref = all_bins[-1].detach()
    weights_ref = all_weights[-1].detach()
    loss = 0
    for bins, weights in zip(all_bins[:-1], all_weights[:-1]):
        loss += loss_interlevel(bins_ref, weights_ref, bins, weights).mean()

    return loss


@torch.cuda.amp.autocast(enabled=False)
def contract(x):
    # x: [..., C]
    shape, C = x.shape[:-1], x.shape[-1]
    x = x.view(-1, C)
    mag, idx = x.abs().max(1, keepdim=True)  # [N, 1], [N, 1]
    scale = 1 / mag.repeat(1, C)
    scale.scatter_(1, idx, (2 - 1 / mag) / mag)
    z = torch.where(mag < 1, x, x * scale)
    return z.view(*shape, C)


@torch.cuda.amp.autocast(enabled=False)
def uncontract(z):
    # z: [..., C]
    shape, C = z.shape[:-1], z.shape[-1]
    z = z.view(-1, C)
    mag, idx = z.abs().max(1, keepdim=True)  # [N, 1], [N, 1]
    scale = 1 / (2 - mag.repeat(1, C)).clamp(min=1e-8)
    scale.scatter_(1, idx, 1 / (2 * mag - mag * mag).clamp(min=1e-8))
    x = torch.where(mag < 1, z, z * scale)
    return x.view(*shape, C)


@torch.cuda.amp.autocast(enabled=False)
def sample_pdf(bins, weights, T, perturb=False):
    # bins: [N, T0+1]
    # weights: [N, T0]
    # return: [N, T]

    N, T0 = weights.shape
    weights = weights + 0.01  # prevent NaNs
    weights_sum = torch.sum(weights, -1, keepdim=True)  # [N, 1]
    pdf = weights / weights_sum
    cdf = torch.cumsum(pdf, -1).clamp(max=1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [N, T+1]

    u = torch.linspace(0.5 / T, 1 - 0.5 / T, steps=T).to(weights.device)
    u = u.expand(N, T)

    if perturb:
        u = u + (torch.rand_like(u) - 0.5) / T

    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)  # [N, t]

    below = torch.clamp(inds - 1, 0, T0)
    above = torch.clamp(inds, 0, T0)

    cdf_g0 = torch.gather(cdf, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g0 = torch.gather(bins, -1, below)
    bins_g1 = torch.gather(bins, -1, above)

    bins_t = torch.clamp(torch.nan_to_num(
        (u - cdf_g0) / (cdf_g1 - cdf_g0)), 0, 1)  # [N, t]
    bins = bins_g0 + bins_t * (bins_g1 - bins_g0)  # [N, t]

    return bins


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.05):
    # rays: [N, 3], [N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [N, 1], far [N, 1]

    tmin = (aabb[:3] - rays_o) / (rays_d + 1e-15)  # [N, 3]
    tmax = (aabb[3:] - rays_o) / (rays_d + 1e-15)
    near = torch.where(tmin < tmax, tmin, tmax).amax(dim=-1, keepdim=True)
    far = torch.where(tmin > tmax, tmin, tmax).amin(dim=-1, keepdim=True)
    # if far < near, means no intersection, set both near and far to inf (1e9 here)
    mask = far < near
    near[mask] = 1e9
    far[mask] = 1e9
    # restrict near to a minimal value
    near = torch.clamp(near, min=min_near)

    return near, far


class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # bound for ray marching (world space)
        self.real_bound = opt.bound

        # bound for grid querying
        if self.opt.contract:
            self.bound = 2
        else:
            self.bound = opt.bound

        self.cascade = 1 + math.ceil(math.log2(self.bound))

        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb_train = torch.FloatTensor(
            [-self.real_bound, -self.real_bound, -self.real_bound, self.real_bound, self.real_bound, self.real_bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

    def forward(self, x, d, **kwargs):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x, **kwargs):
        raise NotImplementedError()

    def update_aabb(self, aabb):
        # aabb: tensor of [6]
        if not torch.is_tensor(aabb):
            aabb = torch.from_numpy(aabb).float()
        self.aabb_train = aabb.clamp(-self.real_bound,
                                     self.real_bound).to(self.aabb_train.device)
        self.aabb_infer = self.aabb_train.clone()
        print(f'[INFO] update_aabb: {self.aabb_train.cpu().numpy().tolist()}')

    def render(self, rays_o, rays_d, staged=False, cam_near_far=None, **kwargs):

        if not staged:
            return self.run(rays_o, rays_d, **kwargs)
        else:  # staged inference
            N = rays_o.shape[0]
            device = rays_o.device

            head = 0
            results = {}
            while head < N:
                tail = min(head + self.opt.max_ray_batch, N)
                if cam_near_far is None:
                    results_ = self.run(
                        rays_o[head:tail], rays_d[head:tail], cam_near_far=None, **kwargs)
                elif cam_near_far.shape[0] == 1:
                    results_ = self.run(
                        rays_o[head:tail], rays_d[head:tail], cam_near_far=cam_near_far, **kwargs)
                else:
                    results_ = self.run(
                        rays_o[head:tail], rays_d[head:tail], cam_near_far=cam_near_far[head:tail], **kwargs)

                for k, v in results_.items():
                    if v is None:
                        continue
                    if torch.is_tensor(v):
                        if k not in results:
                            results[k] = torch.empty(
                                N, *v.shape[1:], device=device)
                        results[k][head:tail] = v
                    else:
                        results[k] = v
                head += self.opt.max_ray_batch

            return results

    def run(self, rays_o, rays_d, bg_color=None, perturb=False, cam_near_far=None, update_proposal=True, return_feats=0, return_mask=0, H=None, W=None, **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]

        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        N = rays_o.shape[0]
        device = rays_o.device

        # pre-calculate near far
        nears, fars = near_far_from_aabb(
            rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, [0]])
            fars = torch.minimum(fars, cam_near_far[:, [1]])

        # mix background color
        if bg_color is None:
            bg_color = 1

        results = {}

        # hierarchical sampling
        if self.training:
            all_bins = []
            all_weights = []

        # sample xyzs using a mixed linear + lindisp function
        def spacing_fn(x): return torch.where(x < 1, x / 2, 1 - 1 / (2 * x))

        def spacing_fn_inv(x): return torch.where(
            x < 0.5, 2 * x, 1 / (2 - 2 * x))

        s_nears = spacing_fn(nears)  # [N, 1]
        s_fars = spacing_fn(fars)  # [N, 1]

        bins = None
        weights = None

        for prop_iter in range(len(self.opt.num_steps)):

            if prop_iter == 0:
                # uniform sampling
                bins = torch.linspace(
                    0, 1, self.opt.num_steps[prop_iter] + 1, device=device).unsqueeze(0)  # [1, T+1]
                bins = bins.expand(N, -1)  # [N, T+1]
                if perturb:
                    bins = bins + (torch.rand_like(bins) - 0.5) / \
                        (self.opt.num_steps[prop_iter])
                    bins = bins.clamp(0, 1)
            else:
                # pdf sampling
                bins = sample_pdf(
                    bins, weights, self.opt.num_steps[prop_iter] + 1, perturb).detach()  # [N, T+1]

            # [N, T+1] in [near, far]
            real_bins = spacing_fn_inv(s_nears * (1 - bins) + s_fars * bins)

            rays_t = (real_bins[..., 1:] + real_bins[..., :-1]) / 2  # [N, T]

            xyzs = rays_o.unsqueeze(
                1) + rays_d.unsqueeze(1) * rays_t.unsqueeze(2)  # [N, T, 3]

            if self.opt.contract:
                xyzs = contract(xyzs)
  
            if prop_iter != len(self.opt.num_steps) - 1:
                # query proposal density
                with torch.set_grad_enabled(update_proposal):
                    sigmas = self.density(xyzs, proposal=prop_iter)['sigma']  # [N, T]
            else:
                # last iter: query nerf
                dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)  # [N, T, 3]
                dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
                outputs = self(xyzs, dirs, save_intermedian_results = self.opt.with_mask and self.opt.mask_mlp_type == 'adaptive')
                sigmas = outputs['sigma']
                colors = outputs['color']
                geo_feat = outputs['geo_feat']
                grid_output = outputs['grid_output']

                if self.opt.with_sam:
                    features = self.s_grid(xyzs, bound=self.bound)

                if return_mask > 0 and (self.opt.mask_mlp_type == 'default' or self.opt.mask_mlp_type == 'lightweight_mask'):
                    masks = self.m_grid(xyzs, bound=self.bound)
                    # masks = colors

            # sigmas to weights
            deltas = (real_bins[..., 1:] - real_bins[..., :-1])  # [N, T]
            deltas_sigmas = deltas * sigmas  # [N, T]

            # opaque background
            if self.opt.background == 'last_sample':
                deltas_sigmas = torch.cat(
                    [deltas_sigmas[..., :-1], torch.full_like(deltas_sigmas[..., -1:], torch.inf)], dim=-1)

            alphas = 1 - torch.exp(-deltas_sigmas)  # [N, T]
            transmittance = torch.cumsum(
                deltas_sigmas[..., :-1], dim=-1)  # [N, T-1]
            transmittance = torch.cat(
                [torch.zeros_like(transmittance[..., :1]), transmittance], dim=-1)  # [N, T]
            transmittance = torch.exp(-transmittance)  # [N, T]

            weights = alphas * transmittance  # [N, T]
            weights.nan_to_num_(0)

            if self.training:
                all_bins.append(bins)
                all_weights.append(weights)

        # composite
        weights_sum = torch.sum(weights, dim=-1)  # [N]

        depth = torch.sum(weights * rays_t, dim=-1)  # [N]


        f_image = torch.sum(weights.unsqueeze(-1) * colors, dim=-2) 
        if self.opt.sum_after_mlp:
            f_colors = self.view_mlp(colors, save_intermedian_results = self.opt.with_mask and self.opt.mask_mlp_type == 'adaptive')
            f_colors_sum = torch.sum(weights.unsqueeze(-1) * f_colors, dim=-2)
            image = torch.sigmoid(f_colors_sum)
        else:
            # f_image = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)  # [N, C]
            image = torch.sigmoid(self.view_mlp(f_image, save_intermedian_results = self.opt.with_mask and self.opt.mask_mlp_type == 'adaptive'))  # [N, 3]

        # extra results
        if self.training and (not self.opt.with_mask and not self.opt.with_sam):
            results['num_points'] = xyzs.shape[0] * xyzs.shape[1]
            results['weights'] = weights

            if self.opt.lambda_proposal > 0 and update_proposal:
                results['proposal_loss'] = proposal_loss(all_bins, all_weights)

            if self.opt.lambda_distort > 0:
                results['distort_loss'] = distort_loss(bins, weights)

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        results['weights_sum'] = weights_sum
        results['depth'] = depth
        results['image'] = image

        if self.opt.with_sam:
            if self.opt.sum_after_mlp:
                if self.opt.sam_use_view_direction:
                    f = torch.cat([features, f_colors, f_colors], dim=-1)
                else:
                    f = torch.cat([features, geo_feat, f_colors],  dim=-1)

                samvit_output = self.samvit_mlp(f, save_intermedian_results = self.opt.with_mask and self.opt.mask_mlp_type == 'adaptive')
                samvit_output = self.norm_layer(samvit_output)
                samvit_output_sum = torch.sum(weights.unsqueeze(-1) * samvit_output, dim=-2)
                if return_feats > 0:
                    samvit_mlp = samvit_output_sum.view(H, W, -1)
            else:
                f_sam = torch.sum(weights.unsqueeze(-1) * features, dim=-2)
                
                if self.opt.sam_use_view_direction:
                    f = torch.cat([f_sam, f_image, image, depth.unsqueeze(-1)], dim=-1)
                else:
                    geo_feat_sum = torch.sum(weights.unsqueeze(-1) * geo_feat, dim=-2) 
                    f = torch.cat([f_sam, geo_feat_sum, image, depth.unsqueeze(-1)], dim=-1)
                
                samvit_mlp = self.samvit_mlp(f)
                # samvit_mlp = self.norm_layer(samvit_mlp)
                if return_feats > 0:
                    samvit_mlp = samvit_mlp.view(H, W, -1)
            if return_feats > 0:
                results['samvit'] = samvit_mlp

        if return_mask > 0:
            if self.opt.mask_mlp_type == 'default':
                m = torch.cat([masks, geo_feat.detach()], dim=-1)
                point_masks = self.mask_mlp(m)
            elif self.opt.mask_mlp_type == 'lightweight_mask':
                m = torch.cat([masks, colors.detach()], dim=-1)
                point_masks = self.mask_mlp(m)
            elif self.opt.mask_mlp_type == 'adaptive':
                if self.opt.adaptive_mlp_type == 'rgb':
                    
                    # architecture 0
                    m = self.mask_mlp[0](grid_output.detach())
                    m = self.mask_mlp[1](torch.cat([self.grid_mlp.intermedian_reuslts[0], m], dim=-1))
                    m = self.mask_mlp[2](torch.cat([self.grid_mlp.intermedian_reuslts[1], m], dim=-1))
                    m = self.mask_mlp[3](torch.cat([self.grid_mlp.intermedian_reuslts[2], m], dim=-1))
        
                    m = self.mask_mlp[4](torch.cat([self.view_mlp.intermedian_reuslts[0], m], dim=-1))
                    m = self.mask_mlp[5](torch.cat([self.view_mlp.intermedian_reuslts[1], m], dim=-1))
                    
                    m = self.mask_mlp[6](m)
                    point_masks = self.mask_mlp[7](m)
                    
                    
                    # architecture 1
                    # m = self.mask_mlp[0](self.grid_mlp.intermedian_reuslts[1])

        
                    # m = self.mask_mlp[1](torch.cat([self.view_mlp.intermedian_reuslts[0], m], dim=-1))
                    # m = self.mask_mlp[2](torch.cat([self.view_mlp.intermedian_reuslts[1], m], dim=-1))
                    
                    # point_masks = self.mask_mlp[3](m)
                    
                elif self.opt.adaptive_mlp_type == 'density':
                    m = self.mask_mlp[0](grid_output.detach())
                    m = self.mask_mlp[1](torch.cat([self.grid_mlp.intermedian_reuslts[0], m], dim=-1))
                    m = self.mask_mlp[2](torch.cat([self.grid_mlp.intermedian_reuslts[1], m], dim=-1))
                    m = self.mask_mlp[3](torch.cat([self.grid_mlp.intermedian_reuslts[2], m], dim=-1))
                    
                    m = self.mask_mlp[4](m)
                    point_masks = self.mask_mlp[5](m)
                    
                    # m = self.mask_mlp[0](grid_output.detach())
                    # m = self.mask_mlp[1](torch.cat([self.grid_mlp.intermedian_reuslts[0], m], dim=-1))
                    # m = self.mask_mlp[2](torch.cat([self.grid_mlp.intermedian_reuslts[1], m], dim=-1))
                    # point_masks = self.mask_mlp[3](torch.cat([self.grid_mlp.intermedian_reuslts[2], m], dim=-1))
                    
                    # m = self.mask_mlp[0](self.grid_mlp.intermedian_reuslts[0])
                    # m = self.mask_mlp[1](torch.cat([self.grid_mlp.intermedian_reuslts[1], m], dim=-1))
                    # point_masks = self.mask_mlp[2](torch.cat([self.grid_mlp.intermedian_reuslts[2], m], dim=-1))
                    
                    
                elif self.opt.adaptive_mlp_type == 'sam':
                    m = self.mask_mlp[0](self.grid_mlp.intermedian_reuslts[0])
                    m = self.mask_mlp[1](torch.cat([self.grid_mlp.intermedian_reuslts[1], m], dim=-1))
                    m = self.mask_mlp[2](torch.cat([self.grid_mlp.intermedian_reuslts[2], m], dim=-1))
  
                    m = self.mask_mlp[3](torch.cat([self.samvit_mlp.intermedian_reuslts[0], m], dim=-1))
                    m = self.mask_mlp[4](torch.cat([self.samvit_mlp.intermedian_reuslts[1], m], dim=-1))
                    m = self.mask_mlp[5](torch.cat([self.samvit_mlp.intermedian_reuslts[2], m], dim=-1))
                    point_masks = self.mask_mlp[6](torch.cat([self.samvit_mlp.intermedian_reuslts[3], m], dim=-1))
                    
            results['instance_mask_logits'] = torch.sum(
                weights.detach().unsqueeze(-1) * point_masks, dim=-2)

            # f_mask = torch.sum(weights.unsqueeze(-1) * masks, dim=-2)
            
            
            
            # m = torch.cat([f_image.detach(), image.detach(), depth.unsqueeze(-1).detach()], dim=-1)
            # point_masks = self.mask_mlp(m)
            # results['instance_mask_logits'] = point_masks

        return results
