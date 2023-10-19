import torch
import argparse
import sys
import os.path as osp

from nerf.utils import *
from segment_anything import build_sam, sam_model_registry_baseline, SamPredictor
import numpy as np

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--fp16', action='store_true',
                        help="use amp mixed precision training")

    parser.add_argument('--with_sam', action='store_true',
                        help="train/test with feats")
    parser.add_argument('--init_ckpt', type=str, default='',
                        help='ckpt to load for with_sam stage')
    parser.add_argument('--sam_ckpt', type=str,
                        default='./pretrained/sam_vit_h_4b8939.pth', help='ckpt to sam-h')
    parser.add_argument('--online_resolution', type=int, default=512,
                        help="NeRF rendering resolution at online distillation")
    parser.add_argument('--sam_use_view_direction', action='store_true',
                        help='use view direction to sam feature')

    parser.add_argument('--cache_size', type=int, default=256,
                        help="online training cache size (on GPU!), <=0 to disable")
    parser.add_argument('--cache_interval', type=int, default=4,
                        help="online training use novel pose per $ iters")

    # testing options
    parser.add_argument('--save_cnt', type=int, default=20,
                        help="save checkpoints for $ times during training")
    parser.add_argument('--eval_cnt', type=int, default=5,
                        help="perform validation for $ times during training")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--camera_traj', type=str, default='interp',
                        help="interp for interpolatxion, circle for circular camera")

    # dataset options
    parser.add_argument('--train_split', type=str,
                        default='train', choices=['train', 'trainval', 'all'])
    parser.add_argument('--test_split', type=str,
                        default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--random_image_batch', action='store_true',
                        help="randomly sample rays from all images per step in training")
    parser.add_argument('--downscale', type=int, default=1,
                        help="downscale training images")
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=-1,
                        help="scale camera location into box[-bound, bound]^3, -1 means automatically determine based on camera poses..")
    parser.add_argument('--offset', type=float, nargs='*',
                        default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--enable_cam_near_far', action='store_true',
                        help="colmap mode: use the sparse points to estimate camera near far per view.")
    parser.add_argument('--enable_cam_center', action='store_true',
                        help="use camera center instead of sparse point center (colmap dataset only)")
    parser.add_argument('--min_near', type=float, default=0.2,
                        help="minimum near distance for camera")
    parser.add_argument('--T_thresh', type=float, default=1e-4,
                        help="minimum transmittance to continue ray marching")

    # training options
    parser.add_argument('--iters', type=int, default=20000,
                        help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2,
                        help="initial learning rate")
    parser.add_argument('--num_steps', type=int, nargs='*', default=[
                        128, 64, 32], help="num steps sampled per ray for each proposal level")
    parser.add_argument('--contract', action='store_true',
                        help="apply spatial contraction as in mip-nerf 360, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--background', type=str, default='last_sample', choices=[
                        'white', 'random', 'last_sample'], help="training background mode")

    parser.add_argument('--max_ray_batch', type=int, default=4096 * 4 ,
                        help="batch size of rays at inference to avoid OOM")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied")

    # batch size related
    parser.add_argument('--num_rays', type=int, default=4096,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--adaptive_num_rays', action='store_true',
                        help="adaptive num rays for more efficient training")
    parser.add_argument('--num_points', type=int, default=2 ** 18,
                        help="target num points for each training step, only work with adaptive num_rays")

    # regularizations
    parser.add_argument('--lambda_entropy', type=float,
                        default=0, help="loss scale")
    parser.add_argument('--lambda_tv', type=float,
                        default=0, help="loss scale")
    parser.add_argument('--lambda_wd', type=float,
                        default=0, help="loss scale")
    parser.add_argument('--lambda_proposal', type=float, default=1,
                        help="loss scale (only for non-cuda-ray mode)")
    parser.add_argument('--lambda_distort', type=float, default=0.02,
                        help="loss scale (only for non-cuda-ray mode)")

    # train mask options
    parser.add_argument('--with_mask', action='store_true', 
                        help="train/test with mask of some object")
    parser.add_argument('--mask_mlp_type', type=str,
                        default='default', choices=['default', 'lightweight_mask', 'adaptive'])
    parser.add_argument('--n_inst', type=int, default=2,
                        help='num of instance')
    parser.add_argument('--label_regularization_weight', type=float,
                        default=0., help="label regularization weight")
    parser.add_argument('--patch_size', type=int, default=1,
                        help='patch size in train sampling')
    parser.add_argument('--pose_jittering', action='store_true')
    parser.add_argument('--mask_folder_name', type=str,
                        default='masks', help="mask folder name")
    
    parser.add_argument('--incoherent_uncertainty_weight', type=float, default=1,
                        help='uncertainty weight applied on the incoherent regions')
    parser.add_argument('--rgb_similarity_loss_weight', type=float, default=0,
                        help="rgb similarity loss weight")
    parser.add_argument('--rgb_similarity_threshold', type=float, default=0.3,
                        help="rgb similarity loss weight") 
    parser.add_argument('--epsilon', type=float, default=0.000001,
                        help="avoid log zero in BCE loss") 
    parser.add_argument('--rgb_similarity_exp_weight', type=float, default=10,
                        help="adjust the number of the similarity function") 
    parser.add_argument('--rgb_similarity_num_sample', type=int, default=1,
                        help='number of sampling points of the similarity function')
    parser.add_argument('--rgb_similarity_iter', type=int, default=-1,
                        help='iteration that will start to use rgb similarity loss')
    parser.add_argument('--rgb_similarity_use_pred_logistics', action='store_true',
                        help='use predicted logistics as reference when encouraging instance similarity')
    
    parser.add_argument('--redundant_instance', type=int, default=0,
                        help='redundant instance output for local contrastive learning')
    parser.add_argument('--sum_after_mlp', action='store_true',
                        help='use point to generate mask')
    parser.add_argument('--adaptive_mlp_type', type=str,
                        default='density', choices=['density', 'rgb', 'sam'])
    parser.add_argument('--use_multi_res', action='store_true',
                        help='use multi-resolution to generate mask')
    
    parser.add_argument('--use_dynamic_incoherent', action='store_true',
                        help='use render masks as incoherent region reference')
    parser.add_argument('--incoherent_update_iter', type=int,
                        default=50, help='use multi-resolution to generate mask')
    parser.add_argument('--incoherent_downsample_scale', type=int,
                        default=1, help='use multi-resolution to generate mask')
     
 
    parser.add_argument('--use_mutli_res', action='store_true',
                        help='use multi resolution in training')
    parser.add_argument('--multi_res_update_iter', type=int,
                        default=100, help='use multi-resolution to generate mask')
    parser.add_argument('--max_multi_res_level', type=int,
                        default=2, help='use multi-resolution to generate mask')
    
    parser.add_argument('--mixed_sampling', action='store_true',
                        help='include local sampling when choosing training points') 
    parser.add_argument('--local_sample_patch_size', type=int, default=16,
                        help='patch size of local sampling')
    parser.add_argument('--num_local_sample', type=int, default=2,
                        help='number of points of local sampling')
    parser.add_argument('--error_map', action='store_true',
                        help='use error map ')
    parser.add_argument('--error_map_size', type=int, default=128,
                        help='size of the error maps')
      
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--return_extra', action='store_true',
                        help='return extra output')

    # evaluation option
    parser.add_argument('--use_point', action='store_true',
                        help='use point to generate mask')
    parser.add_argument('--val_type', type=str, default='default', choices=['default', 'val_all', 'val_split'],
                        help='evaluate all images')

    # render mask options
    parser.add_argument('--render_mask_type', type=str, default='heatmap', choices=['mask', 'composition', 'heatmap'],
                        help='type of mask that the system will render \n mask: output mask only \n composition: '
                        'output mask composed with rgb image \n heatmap: output heatmap')
    parser.add_argument('--render_mask_instance_id', type=int, default=0,
                        help='determine which instance it will choose to render')

    # GUI options
    parser.add_argument('--vis_pose', action='store_true',
                        help="visualize the poses")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=512, help="GUI width")
    parser.add_argument('--H', type=int, default=512, help="GUI height")
    parser.add_argument('--radius', type=float, default=0.5,
                        help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60,
                        help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1,
                        help="GUI rendering max sample per pixel")

    parser.add_argument('--data_type', type=str, default='mip',
                        choices=['mip', 'lerf', 'llff', '3dfront', 'ctr', 'pano', 'lift'], help="dataset type")
    
    opt = parser.parse_args()

    opt.fp16 = False
    opt.bound = 128  # large enough
    opt.preload = True  # unset if CUDA OOM
    opt.contract = True
    opt.adaptive_num_rays = True
    # opt.random_image_batch = True

    from nerf.colmap_provider import ColmapDataset
    from nerf.lerf_provider import LERFDataset

    dataset_dict = {
        'mip': ColmapDataset
    }
    NeRFDataset = ColmapDataset

    seed_everything(opt.seed)

    from nerf.network import NeRFNetwork

    if opt.with_mask:
        if opt.n_inst > 1:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')   
        else:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  
        
    else: 
        criterion = torch.nn.MSELoss(reduction='none')
    # criterion = torch.nn.SmoothL1Loss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt).to(device)

    if (opt.with_sam or opt.with_mask) and opt.init_ckpt != '':
        # load pretrained checkpoint of rgbd
        model_dict = torch.load(opt.init_ckpt, map_location=device)['model']
        model.load_state_dict(model_dict, strict=False)
        # freeze rgbd params
        for k, v in model.named_parameters():
            if k in model_dict:
                v.requires_grad = False

    if opt.with_sam:
        
        sam_predictor = SamPredictor(
            sam_model_registry_baseline["vit_h"](checkpoint=opt.sam_ckpt).eval().to(device))
    else:
        sam_predictor = None

    if opt.test:
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace,
                          criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt, sam_predictor=sam_predictor)

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(opt, device=device, type=opt.test_split)
            test_loader.training = False
            test_loader = test_loader.dataloader()

            if opt.test_split != 'test':
                trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]  # set up metrics
                if opt.with_mask:
                    trainer.metrics = [MeanIoUMeter()]
                # blender has gt, so evaluate it.
                trainer.evaluate(test_loader)

            trainer.test(test_loader, write_video=False)  # test and save video

    else:

        optimizer = torch.optim.Adam(model.get_params(opt.lr), eps=1e-15)

        train_loader = NeRFDataset(
            opt, device=device, type=opt.train_split).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        # save ~50 times during the training
        save_interval = max(1, max_epoch // max(1, opt.save_cnt))
        eval_interval = max(1, max_epoch // max(1, opt.eval_cnt))
        print(
            f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}.')

        # colmap can estimate a more compact AABB
        if not opt.contract and opt.data_format == 'colmap':
            model.update_aabb(train_loader._data.pts_aabb)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, 
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval, 
                          save_interval=save_interval, sam_predictor=sam_predictor)

        if opt.use_wandb:
            wandb.init(config=opt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        else:
            valid_loader = NeRFDataset(
                opt, device=device, type='val').dataloader()

            trainer.metrics = [PSNRMeter(),]
            if opt.with_mask:
                trainer.metrics = [MeanIoUMeter()]
            trainer.train(train_loader, valid_loader, max_epoch)

            # last validation
            trainer.metrics = [PSNRMeter(), SSIMMeter(),LPIPSMeter(device=device)]
            if opt.with_mask:
                trainer.metrics = [MeanIoUMeter()]
            trainer.evaluate(valid_loader)

            # # also test
            # test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            # if test_loader.has_gt:
            #     trainer.evaluate(test_loader) # blender has gt, so evaluate it.

            # trainer.test(test_loader, write_video=True) # test and save video
