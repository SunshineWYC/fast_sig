import os
import sys
import numpy as np
import torch
import time
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.camera_utils import update_pose
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, build_scaling_rotation
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.schedule_utils import TrainingScheduler
from utils.pose_refine_utils import export_refined_colmap_model
from utils.lib_bilagrid import BilateralGrid, slice as bilgrid_slice
import json


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def training(dataset, opt, pipe, debug_from, log_file=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    
    # Init Scene and Gaussains
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, pipe, None, shuffle=False)
    gaussians = scene.gaussians
    torch.cuda.empty_cache()

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Init DashGaussian scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians,
                                  [cam.original_image for cam in scene.getTrainCameras()])

    bil_grids = None
    bilgrid_optimizer = None
    bilgrid_lr_fn = None
    bilgrid_cam_idx_map = None
    if opt.use_bilgrid_3d:
        # bilateral grid setup, parameters and optimizer
        train_cameras = scene.getTrainCameras()
        bilgrid_cam_idx_map = {cam.image_name: idx for idx, cam in enumerate(train_cameras)}
        bil_grids = BilateralGrid(
            num=len(train_cameras),
            grid_X=opt.bilgrid_width,
            grid_Y=opt.bilgrid_height,
            grid_W=opt.bilgrid_depth,
        ).to("cuda")
        bilgrid_optimizer = torch.optim.Adam(
            bil_grids.parameters(),
            lr=opt.bilgrid_lr_init,
            eps=1e-15,
        )
        bilgrid_lr_fn = get_expon_lr_func(
            lr_init=opt.bilgrid_lr_init,
            lr_final=opt.bilgrid_lr_final,
            lr_delay_steps=opt.bilgrid_lr_delay_steps,
            lr_delay_mult=opt.bilgrid_lr_delay_mult,
            max_steps=opt.iterations,
        )

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.use_pose_optimization:
        for cam in scene.getTrainCameras():
            cam.init_pose_optimizer(opt.pose_rot_lr, opt.pose_trans_lr)
    
    render_scale = scheduler.get_res_scale(1)
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        xyz_lr = gaussians.update_learning_rate(iteration)
        render_scale = scheduler.get_res_scale(iteration)

        # update bilateral grid learning rate
        if bilgrid_optimizer is not None:
            bilgrid_lr = bilgrid_lr_fn(iteration) * opt.bilgrid_lr_scale
            for param_group in bilgrid_optimizer.param_groups:
                param_group["lr"] = bilgrid_lr


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        pose_active = (
            opt.use_pose_optimization
            and iteration > opt.pose_update_from
            and iteration < opt.pose_update_until
        )

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Rescale GT image for DashGaussian
        gt_image = viewpoint_cam.original_image.cuda()
        if render_scale > 1:
            gt_image = torch.nn.functional.interpolate(gt_image[None], scale_factor=1/render_scale, mode="bilinear", recompute_scale_factor=True, antialias=True)[0]

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            render_size=gt_image.shape[-2:],
            cam_rot_delta=viewpoint_cam.cam_rot_delta if pose_active else None,
            cam_trans_delta=viewpoint_cam.cam_trans_delta if pose_active else None,
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # apply bilateral grid correction
        if bil_grids is not None:
            h, w = image.shape[-2:]
            device = image.device
            dtype = image.dtype

            y = (torch.arange(h, device=device, dtype=dtype) + 0.5) / float(h)
            x = (torch.arange(w, device=device, dtype=dtype) + 0.5) / float(w)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            pix_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

            rgb_flat = image.permute(1, 2, 0).reshape(-1, 3)
            cam_idx = bilgrid_cam_idx_map[viewpoint_cam.image_name]
            cam_idx = torch.full(
                (pix_xy.shape[0], 1),
                fill_value=cam_idx,
                device=device,
                dtype=torch.long,
            )

            bilgrid_results = bilgrid_slice(bil_grids, pix_xy, rgb_flat, cam_idx)
            image = bilgrid_results["rgb"].reshape(h, w, 3).permute(2, 0, 1).contiguous()

        # Loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if dataset.use_depth_supervision and depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            
            if render_scale > 1:
                mono_invdepth = torch.nn.functional.interpolate(mono_invdepth[None], scale_factor=1/render_scale, mode="bilinear", recompute_scale_factor=True, antialias=True)[0, 0]  

            Ll1depth_pure = torch.abs(invDepth  - mono_invdepth).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure * 0.3
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
        
        # MCMC loss
        if opt.use_mcmc:
            loss = loss + opt.opacity_reg * torch.abs(gaussians.get_opacity).mean()
            loss = loss + opt.scale_reg * torch.abs(gaussians.get_scaling).mean()
        
        # bilateral grid regularization
        if bil_grids is not None and opt.bilgrid_tv_loss_mult > 0:
            loss = loss + opt.bilgrid_tv_loss_mult * bil_grids.tv_loss()

        loss.backward()

        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "Ll1depth": f"{ema_Ll1depth_for_log:.{4}f}", "N_GS": f"{gaussians._scaling.shape[0]}", "N_MAX": f"{scheduler.max_n_gaussian}", "R": f"{render_scale}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            # MCMC densification
            if (
                opt.use_mcmc
                and iteration < opt.densify_until_iter
                and iteration > opt.densify_from_iter
                and iteration % opt.densification_interval == 0
            ):
                dead_mask = (gaussians.get_opacity <= opt.min_opacity).squeeze(-1)
                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(cap_max=pipe.max_n_gaussian, new_gaussian_ratio=opt.new_gaussian_ratio)

            # Optimizer step
            if iteration < opt.iterations:
                if dataset.train_test_exp:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                # bilate grid optimizer update
                if bilgrid_optimizer is not None:
                    bilgrid_optimizer.step()
                    bilgrid_optimizer.zero_grad(set_to_none=True)

                # MCMC noise SGLD addition
                if opt.use_mcmc:
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))

                    noise = (
                        torch.randn_like(gaussians._xyz)
                        * op_sigmoid(1 - gaussians.get_opacity)
                        * opt.noise_lr
                        * xyz_lr
                    )
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

                if pose_active and viewpoint_cam.pose_optimizer is not None:
                    viewpoint_cam.pose_optimizer.step()
                    viewpoint_cam.pose_optimizer.zero_grad(set_to_none=True)
            
            # Update camera poses
            if (
                opt.use_pose_optimization
                and iteration > opt.pose_update_from
                and iteration % opt.pose_update_interval == 0
                and iteration < opt.pose_update_until
            ):
                for view in scene.getTrainCameras():
                    update_pose(view)
    
    if opt.use_pose_optimization:
        for view in scene.getTrainCameras():
            update_pose(view)

    point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}/point_cloud.ply".format(iteration))
    scene.gaussians.save_ply(point_cloud_path)

    with open(os.path.join(scene.model_path, "TRAIN_INFO"), "w+") as f:
        f.write("GS Number: {}\n".format(gaussians._scaling.shape[0]))

    pose_refined_dir = os.path.join(scene.model_path, "pose_refined")
    source_sparse_dir = os.path.join(dataset.source_path, "sparse", "0")
    export_refined_colmap_model(
        source_sparse_dir=source_sparse_dir,
        out_dir=pose_refined_dir,
        refined_train_cameras=scene.getTrainCameras(),
    )
    scene.gaussians.save_ply(os.path.join(pose_refined_dir, "gs_points.ply"))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--log_file", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.debug_from, args.log_file)

    # All done
    print("\nTraining complete.")
