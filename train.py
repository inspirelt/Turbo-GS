#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
import time
from PIL import Image
import torchvision.transforms.functional as tf
import json
from pathlib import Path
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Add wandb import
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

def score_func(view, gaussians, pipeline, background, scores):

    img_scores = torch.zeros_like(scores)
    img_scores.requires_grad = True

    image = render(view, gaussians, pipeline, background,
                   scores=img_scores)['render']
    
    image.sum().backward()

    scores += img_scores.grad


def prune(scene, gaussians, pipe, background, prune_ratio):
    with torch.enable_grad():
        pbar = tqdm(
            total=len(scene.getTrainCameras()),
            desc='Computing Pruning Scores')
        scores = torch.zeros_like(gaussians.get_opacity.detach())
        for view in scene.getTrainCameras()[::5]:
            score_func(view, gaussians, pipe, background,
                scores)
            pbar.update(1)
        pbar.close()

    gaussians.prune_gaussians(prune_ratio, scores)
    

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
    # Use CUDA when available, otherwise fall back to CPU
    _lpips_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lpips_fn = lpips.LPIPS(net='vgg').to(_lpips_device)
except Exception:
    LPIPS_AVAILABLE = False
    lpips_fn = None
    print('LPIPS not available: LPIPS metrics will be skipped')

def saveRuntimeCode(dst: str) -> None:
    import shutil, pathlib
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def write_profile_timing_file(model_path, timing_stats):
    os.makedirs(model_path, exist_ok=True)
    total_s = max(timing_stats["iter_total_s"], 1e-12)
    timing_file = os.path.join(model_path, "profile_time.txt")
    with open(timing_file, "w") as f:
        f.write("=== Time Profiling Summary ===\n")
        f.write(f"sample_and_render_s: {timing_stats['sample_and_render_s']:.4f} ({100.0 * timing_stats['sample_and_render_s'] / total_s:.2f}%)\n")
        f.write(f"backward_s: {timing_stats['backward_s']:.4f} ({100.0 * timing_stats['backward_s'] / total_s:.2f}%)\n")
        f.write(f"report_s: {timing_stats['report_s']:.4f} ({100.0 * timing_stats['report_s'] / total_s:.2f}%)\n")
        f.write(f"densify_s: {timing_stats['densify_s']:.4f} ({100.0 * timing_stats['densify_s'] / total_s:.2f}%)\n")
        f.write(f"optimizer_s: {timing_stats['optimizer_s']:.4f} ({100.0 * timing_stats['optimizer_s'] / total_s:.2f}%)\n")
        f.write(f"gui_s: {timing_stats['gui_s']:.4f} ({100.0 * timing_stats['gui_s'] / total_s:.2f}%)\n")
        f.write(f"iter_total_s: {timing_stats['iter_total_s']:.4f}\n")
        measured_s = (
            timing_stats['sample_and_render_s']
            + timing_stats['backward_s']
            + timing_stats['report_s']
            + timing_stats['densify_s']
            + timing_stats['optimizer_s']
            + timing_stats['gui_s']
        )
        f.write(f"unaccounted_s: {max(timing_stats['iter_total_s'] - measured_s, 0.0):.4f}\n")
        f.write(f"peak_points: {timing_stats['peak_points']}\n")
        f.write(f"final_points: {timing_stats['final_points']}\n")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, dilation, scale, debug_from, logger, ssim_ratio=1.0, batch_from=0, disable_dilation=100_000_000_000, disable_io=False, profile_time=False):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer, wandb_run = prepare_output_and_logger(dataset, disable_io=disable_io)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, resolution_scales=[1.0])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    stages = [0,0] # [7000, 15000]
    cur_scale = scale
    viewpoint_stack = scene.getTrainCameras(cur_scale).copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", disable=profile_time)
    first_iter += 1

    if dilation > 1:
        # If dilation is enabled, we prepare a list of pixel offsets to render different subsets of pixels in each iteration. This allows for faster training in the early stages by rendering fewer pixels and then gradually increasing the density as training progresses.
        offsets = [(i, j) for i in range(dilation) for j in range(dilation)]
    else:
        #If dilation is not enabled, we just use a single offset of (0,0) which means rendering all pixels without skipping.
        offsets = [(0,0)]

    initial_number = gaussians.get_xyz.shape[0]

    sub_iter = 4
    timing_stats = {
        "sample_and_render_s": 0.0,
        "backward_s": 0.0,
        "report_s": 0.0,
        "densify_s": 0.0,
        "optimizer_s": 0.0,
        "gui_s": 0.0,
        "iter_total_s": 0.0,
        "peak_points": int(gaussians.get_xyz.shape[0]),
        "final_points": int(gaussians.get_xyz.shape[0]),
    }

    for iteration in range(first_iter, opt.iterations + 1):
        iter_wall_start = time.perf_counter()

        if iteration > disable_dilation:
            #Disable dilation after a certain number of iterations, to get sharper results towards the end of training.
            dilation = 1
            offsets = [(0,0)]

        gui_start = time.perf_counter()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=use_sparse_adam)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                try:
                    logger.exception("network_gui.receive() error; closing connection: %s", e)
                except Exception:
                    # If logging itself fails for any reason, print the error.
                    print(f"network_gui.receive() error: {e}")
                network_gui.conn = None
        timing_stats["gui_s"] += time.perf_counter() - gui_start

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        total_sub_loss = 0.0
        if iteration > opt.iterations-batch_from:
            sub_iterations = 4
        else:
            sub_iterations = 1

        render_start = time.perf_counter()
        for sub_iteration in range(sub_iterations):
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras(cur_scale).copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)

            offset = offsets[iteration%len(offsets)]
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, dilation_factor=dilation, offset=offset, use_trained_exp=dataset.train_test_exp, separate_sh=use_sparse_adam)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gaussian_color = render_pkg["color"]

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                alpha_mask = alpha_mask[:, offset[0]::dilation, offset[1]::dilation]
                # image *= alpha_mask
                image = image[:, :min(image.shape[1], alpha_mask.shape[1]), :min(image.shape[2], alpha_mask.shape[2])] * alpha_mask[:, :min(image.shape[1], alpha_mask.shape[1]), :min(image.shape[2], alpha_mask.shape[2])]
    
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            gt_image = gt_image[:, offset[0]::dilation, offset[1]::dilation]
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
    
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) * ssim_ratio
    
            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
    
                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            total_sub_loss += loss

        timing_stats["sample_and_render_s"] += time.perf_counter() - render_start

        backward_start = time.perf_counter()
        total_sub_loss.backward()
        timing_stats["backward_s"] += time.perf_counter() - backward_start

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            # Skip reporting and console updates in profile mode.
            if not profile_time:
                # Log to wandb more frequently
                if WANDB_FOUND and wandb_run and iteration % 10 == 0:
                    wandb_run.log({
                        "train/loss": loss.item(),
                        "train/l1_loss": Ll1.item(),
                        "train/depth_loss": Ll1depth,
                        "train/ssim": ssim_value.item() if not isinstance(ssim_value, float) else ssim_value,
                        "train/ema_loss": ema_loss_for_log,
                        "train/ema_depth_loss": ema_Ll1depth_for_log,
                        "iteration_time_ms": iter_start.elapsed_time(iter_end),
                        "total_points": gaussians.get_xyz.shape[0]
                    }, step=iteration)

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                    elapsed_time = progress_bar.format_dict['elapsed']
                    logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
                    logger.info(f"Total points: {gaussians.get_xyz.shape[0]}")

                # Log and save
                report_start = time.perf_counter()
                training_report(tb_writer, wandb_run, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, fused_ssim, lpips_fn, (pipe, background, use_sparse_adam, dataset.train_test_exp), dataset.train_test_exp, logger)
                timing_stats["report_s"] += time.perf_counter() - report_start
            if (not disable_io) and (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
                # # Log model checkpoint to wandb
                # if WANDB_FOUND and wandb_run:
                #     model_checkpoint_path = f"{scene.model_path}/point_cloud/iteration_{iteration}/point_cloud.ply"
                #     if os.path.exists(model_checkpoint_path):
                #         wandb_run.log({"point_cloud": wandb.Artifact(
                #             name=f"point_cloud_{iteration}", 
                #             type="point_cloud",
                #             description=f"3D Gaussian point cloud at iteration {iteration}"
                #         )})

            # Densification
            densify_start = time.perf_counter()
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, gaussian_color, visibility_filter, iteration)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if args.max_cap == -1:
                        # If max_cap is -1, we use a very large budget.
                        budget = 100_000_000_000
                    else:
                        # We calculate the densification budget based on the initial number of Gaussians, the maximum cap, the current iteration, and the densification schedule. The 'adaptive_power_th' mode allows for a gradual increase in the number of Gaussians added during densification.
                        budget = gaussians.calculate_budget(initial_number, args.max_cap, opt.densify_until_iter-opt.densify_from_iter, iteration-opt.densify_from_iter, mode='adaptive_power_th')
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, budget, args.color_ratio)
                
                # --- Soft Pruning ---
                if (iteration >= opt.prune_from_iter) and \
                    (iteration < opt.prune_until_iter) and \
                    (iteration % opt.prune_interval == 0):

                    prune(scene, gaussians, pipe, background,
                        opt.densify_prune_ratio)
  
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            timing_stats["densify_s"] += time.perf_counter() - densify_start

            # Optimizer step
            optim_start = time.perf_counter()
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
            timing_stats["optimizer_s"] += time.perf_counter() - optim_start

            if (not disable_io) and (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                torch.save((gaussians.capture(), iteration), checkpoint_path)
            current_points = int(gaussians.get_xyz.shape[0])
            timing_stats["peak_points"] = max(timing_stats["peak_points"], current_points)
            timing_stats["final_points"] = current_points
            timing_stats["iter_total_s"] += time.perf_counter() - iter_wall_start
                
                # # Log checkpoint to wandb
                # if WANDB_FOUND and wandb_run:
                #     artifact = wandb.Artifact(
                #         name=f"checkpoint_{iteration}", 
                #         type="model",
                #         description=f"Model checkpoint at iteration {iteration}"
                #     )
                #     artifact.add_file(checkpoint_path)
                #     wandb_run.log_artifact(artifact)

    if profile_time:
        scene.save(opt.iterations)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        write_profile_timing_file(scene.model_path, timing_stats)

def prepare_output_and_logger(args, disable_io=False):    
    """Prepare output directory, TensorBoard writer and optional Weights & Biases run.

    Returns a tuple `(tb_writer, wandb_run)` where `tb_writer` is a
    `torch.utils.tensorboard.SummaryWriter` when available, otherwise `None`.
    `wandb_run` is an initialized W&B run when available, otherwise `None`.
    """
    if disable_io:
        print("I/O disabled: skipping cfg_args, TensorBoard, and Weights & Biases logging")
        return None, None

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    # Initialize wandb
    wandb_run = None
    if WANDB_FOUND:
        # Extract experiment name from model_path
        experiment_name = os.path.basename(os.path.normpath(args.model_path))
        # Initialize wandb
        wandb_run = wandb.init(
            project=f"3dgs-{args.model_path.split('/')[-2]}",
            name=experiment_name,
            config=vars(args),
            dir=args.model_path,
            # Set to False to disable syncing to cloud if you want to test locally first
            # mode="disabled" 
        )
        print("Weights & Biases logging initialized")
    else:
        print("Weights & Biases not available: not logging to wandb")
        
    return tb_writer, wandb_run

def training_report(tb_writer, wandb_run, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, ssim_fn, lpips_fn, renderArgs, train_test_exp, logger):
    """Log training metrics and run evaluation renders at test intervals.

    This writes scalar metrics to TensorBoard when `tb_writer` is provided
    and logs to Weights & Biases when `wandb_run` is available. At
    `testing_iterations` it renders validation views and computes metrics.
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                # lpips_test = 0.0
                
                # Collect images for wandb logging
                wandb_images = []
                
                for idx, viewpoint in enumerate(config['cameras']):
                    # Unpack renderArgs properly
                    pipe, background, separate_sh, train_test_exp_val = renderArgs
                    render_result = renderFunc(viewpoint, scene.gaussians, pipe, background, 
                                               use_trained_exp=train_test_exp_val, separate_sh=separate_sh)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if train_test_exp_val:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    # Prepare images for wandb (first 5 only)
                    if WANDB_FOUND and wandb_run and idx < 5:
                        # Convert to numpy for wandb
                        render_np = image.permute(1, 2, 0).cpu().numpy()
                        gt_np = gt_image.permute(1, 2, 0).cpu().numpy()
                        
                        # Add to list for logging
                        # wandb_images.append(wandb.Image(
                        #     render_np, 
                        #     caption=f"{config['name']}_view_{viewpoint.image_name}_render"
                        # ))
                        # if iteration == testing_iterations[0]:
                        #     wandb_images.append(wandb.Image(
                        #         gt_np, 
                        #         caption=f"{config['name']}_view_{viewpoint.image_name}_ground_truth"
                        #     ))
                    
                    # Calculate metrics

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    # with torch.no_grad():
                    #     lpips_test += lpips_fn(image, gt_image).mean().double()
                    # torch.cuda.empty_cache()
                
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                
                # Log metrics and images to wandb
                if WANDB_FOUND and wandb_run:
                    wandb_run.log({
                        f"{config['name']}/l1_loss": l1_test,
                        f"{config['name']}/psnr": psnr_test,
                        f"{config['name']}/ssim": ssim_test,
                        # f"{config['name']}/lpips": lpips_test,
                        # f"{config['name']}_images": wandb_images if wandb_images else None
                    }, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            
        # Log histogram to wandb
        if WANDB_FOUND and wandb_run:
            opacity_hist = scene.gaussians.get_opacity.cpu().numpy()
            wandb_run.log({
                "scene/opacity_histogram": wandb.Histogram(opacity_hist),
                "total_points": scene.gaussians.get_xyz.shape[0]
            }, step=iteration)
            
        torch.cuda.empty_cache()

def get_logger(path, disable_io=False, silent=False):
    """Create and return a logger that writes to console and `outputs.log`.

    The logger level is INFO and the log file is written under `path`.
    """
    import logging

    logger = logging.getLogger("train_disabledilation_profile")
    logger.setLevel(logging.INFO) 
    logger.handlers = []
    logger.propagate = False

    if silent:
        return logger

    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    controlshow.setFormatter(formatter)

    logger.addHandler(controlshow)

    if not disable_io:
        fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
        fileinfo.setLevel(logging.INFO)
        fileinfo.setFormatter(formatter)
        logger.addHandler(fileinfo)

    return logger

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, quiet=False):
    """Render a set of `views` and save rendered images and ground-truths.

    Parameters
    - `model_path`, `name`, `iteration`: destination folder structure.
    - `views`: iterable of camera/view objects to render.
    - `gaussians`, `pipeline`, `background`: rendering inputs.
    - `train_test_exp`: boolean indicating experiment split handling.
    - `separate_sh`: whether to use separate spherical harmonics rendering.
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    import time
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", disable=quiet)):
        torch.cuda.synchronize();t0 = time.time()
        
        # If needed for debugging/high-resolution stitching,
        # Use the dilation_factor and offset parameters of the render function to render different pixel subsets and stitch together.
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]


        torch.cuda.synchronize();t1 = time.time()
        t_list.append(t1-t0)
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
    import numpy as np
    avg_time = np.array(t_list[5:]).mean()
    if not quiet:
        print(f'avg test time: {avg_time:.5f}')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, separate_sh: bool, logger=None, quiet=False):
    """Helper to render test set for a loaded scene at `iteration`.

    This loads the scene from `dataset.model_path` and calls `render_set`.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, quiet=quiet)

def readImages(renders_dir, gt_dir):
    """Read PNG/JPEG images from `renders_dir` and `gt_dir` and return tensors.

    Returns `(renders, gts, image_names)` where `renders` and `gts` are lists
    of 4D tensors (batch, 3, H, W) on CUDA and `image_names` are filenames.
    """
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, logger=None, quiet=False):
    """Evaluate rendered outputs under `model_paths/test/*` computing metrics.

    Produces `results.json` and `per_view.json` in each `model_paths` folder.
    """

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    # logger.info("")

    for scene_dir in [model_paths]:
        try:
            # logger.info("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                # logger.info("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}
                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"

                

                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress", disable=quiet):
                    # try:
                        with torch.no_grad():
                            ssims.append(ssim(renders[idx], gts[idx]))
                            psnrs.append(psnr(renders[idx], gts[idx]))
                            # Compute LPIPS only if available; otherwise append zero
                            if lpips_fn is not None:
                                try:
                                    lpip_val = lpips_fn(renders[idx], gts[idx])
                                except Exception:
                                    lpip_val = torch.tensor(0.0)
                            else:
                                lpip_val = torch.tensor(0.0)
                            lpipss.append(lpip_val)
                        torch.cuda.empty_cache()

                if not quiet and logger:
                    logger.info("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    logger.info("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    logger.info("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                # logger.info("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            if logger and not quiet:
                logger.info("Unable to compute metrics for model", scene_dir)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    

    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--max_cap", type=int, default=-1) # -1 means no scheduler
    parser.add_argument("--batch_from", type=int, default=0) # -1 means no scheduler
    parser.add_argument("--color_ratio", type=float, default=1) # 1 means not use
    parser.add_argument("--ssim_ratio", type=float, default=1.0) # 1 means not use
    parser.add_argument("--disable_dilation", type=int, default=1000000) # 1 means not use

    
    # Add wandb-specific arguments
    parser.add_argument("--wandb_project", type=str, default="3d-gaussian-splatting", 
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                       help="Weights & Biases entity (team) name")
    parser.add_argument("--disable_wandb", action="store_true", default=False,
                       help="Disable Weights & Biases logging")
    parser.add_argument("--disable_io", action="store_true", default=False,
                       help="Disable script-side output IO (checkpoints, renders, metrics files, TB/W&B, logs to file)")
    parser.add_argument("--profile_time", action="store_true", default=False,
                       help="Print aggregated runtime profile for training stages")
    
    args = parser.parse_args(sys.argv[1:])
    if args.disable_io and not args.profile_time:
        args.disable_viewer = True
        args.disable_wandb = True
        args.test_iterations = []
        args.save_iterations = []
        args.checkpoint_iterations = []

    args.save_iterations.append(args.iterations)
    
    model_path = args.model_path
    if ((not args.disable_io) or args.profile_time) and model_path:
        os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path if model_path else ".", disable_io=args.disable_io or args.profile_time, silent=args.profile_time)

    if not args.profile_time:
        logger.info(f'args: {args}')

    if not args.profile_time:
        logger.info("Optimizing " + args.model_path)

    # (optional) source backup was disabled to avoid copying large trees
    # during experiments. Re-enable the `saveRuntimeCode(...)` call above
    # if you want to keep a copy of the runtime source with each run.

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Initialize wandb if available and not disabled
    if WANDB_FOUND and not args.disable_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
    elif args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.dilation, args.scale, args.debug_from, logger, args.ssim_ratio, args.batch_from, args.disable_dilation, disable_io=args.disable_io or args.profile_time, profile_time=args.profile_time)

    # Close wandb run if it was initialized
    if WANDB_FOUND and 'wandb' in sys.modules and wandb.run is not None:
        wandb.finish()

    # All done
    logger.info("\nTraining complete.")


    # rendering
    if (not args.disable_io) or args.profile_time:
        if not args.profile_time:
            logger.info(f'\nStarting Rendering~')
        render_sets(lp.extract(args), -1, pp.extract(args), separate_sh=False, logger=logger, quiet=args.profile_time)
        if not args.profile_time:
            logger.info("\nRendering complete.")

            # calc metrics
        if not args.profile_time:
            logger.info("\n Starting evaluation...")
        evaluate(args.model_path, logger=logger, quiet=args.profile_time)
        if not args.profile_time:
            logger.info("\nEvaluating complete.")