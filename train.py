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
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss 
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from sklearn.decomposition import PCA
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F
from models.networks import CNN_decoder


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, only_add_features=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1 if only_add_features else None)
    if only_add_features:
        first_iter = scene.loaded_iter
        print(f"Only adding features to existing 3DGS model from iteration {first_iter}")
        gaussians.set_gaussian_opt_enabled(False)

    # 2D semantic feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()
    feature_out_dim = gt_feature_map.shape[0]

    
    # speed up
    if dataset.speedup:
        feature_in_dim = int(feature_out_dim/4)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)

    gaussians.training_setup(opt)
    if checkpoint and not only_add_features:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        feature_map, image, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        mask = viewpoint_cam.mask.cuda(non_blocking=True)
        gt_image = viewpoint_cam.original_image.cuda(non_blocking=True)

        ignore_mask = mask[0] == 0
        image[:, ignore_mask] = gt_image[:, ignore_mask]

        Ll1 = l1_loss(image, gt_image)

        gt_feature_map = viewpoint_cam.semantic_feature.cuda(non_blocking=True)
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) 
        feature_ignore_mask = F.interpolate(mask.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='nearest').squeeze(0)
        feature_ignore_mask = feature_ignore_mask[0] == 0
        if dataset.speedup:
            feature_map = cnn_decoder(feature_map)
        feature_map[:, feature_ignore_mask] = gt_feature_map[:, feature_ignore_mask]

        Ll1_feature = l1_loss(feature_map, gt_feature_map) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 1.0 * Ll1_feature 

        # Weight loss by the number of valid pixels in the mask, i.e., more complete images have more weight
        # loss_scaling = mask.sum() / torch.numel(mask)
        # loss = loss * loss_scaling

        # Anistropic regularization from PhysGaussian
        # Value set based on https://github.com/XPandora/PhysGaussian/issues/18#issuecomment-2045885246
        reg_min_max_axis_ratio = 3
        gaussians_scaling = gaussians.get_scaling
        gaussians_scaling_ratio = torch.max(gaussians_scaling, axis=1).values / torch.min(gaussians_scaling, axis=1).values.clamp(min=1e-8)
        reg_loss = F.relu(gaussians_scaling_ratio - reg_min_max_axis_ratio).mean()
        # reg_loss = torch.tensor(0.0).to(loss)

        (loss + reg_loss).backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), reg_loss.item()) 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print("\n[ITER {}] Saving feature decoder ckpt".format(iteration))
                if dataset.speedup:
                    torch.save(cnn_decoder.state_dict(), scene.model_path + "/decoder_chkpnt" + str(iteration) + ".pth")
  

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if dataset.speedup:
                    cnn_decoder_optimizer.step()
                    cnn_decoder_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
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
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, reg_loss=0.0):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/reg_loss', reg_loss, iteration)
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
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        # Apply PCA to reduce feature dimensions to 3 for RGB visualization using scikit-learn
                        feature_map = render_pkg["feature_map"]  # (256, H, W)
                        # Move to CPU and convert to numpy for sklearn
                        feature_map_np = feature_map.permute(1, 2, 0).cpu().numpy()
                        h, w, c = feature_map_np.shape
                        feature_map_reshaped = feature_map_np.reshape(-1, c)
                        
                        # Apply PCA to get 3 components
                        pca = PCA(n_components=3)
                        pca_result = pca.fit_transform(feature_map_reshaped)
                        
                        # Normalize to [0, 1] for RGB visualization
                        pca_min = pca_result.min(axis=0, keepdims=True)
                        pca_max = pca_result.max(axis=0, keepdims=True)
                        pca_normalized = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)
                        
                        # Reshape back to image dimensions and convert back to torch tensor
                        pca_rgb = pca_normalized.reshape(h, w, 3)
                        pca_rgb_tensor = torch.from_numpy(pca_rgb).permute(2, 0, 1).float().to(feature_map.device)
                        
                        # Add both original feature map and PCA visualization to tensorboard
                        tb_writer.add_images(config['name'] + "_view_{}/feature_map_pca".format(viewpoint.image_name), pca_rgb_tensor.unsqueeze(0), global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    default_iters = ",".join([str(it) for it in range(0, op.iterations, 1000)])
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=str, default=default_iters)
    parser.add_argument("--save_iterations", type=str, default=default_iters)
    parser.add_argument("--checkpoint_iterations", type=str, default=default_iters)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--only_add_features", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations = {*list(map(int, args.save_iterations.split(","))), args.iterations}
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        testing_iterations=list(map(int, args.test_iterations.split(","))),
        saving_iterations=args.save_iterations,
        checkpoint_iterations=list(map(int, args.checkpoint_iterations.split(","))),
        checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
        only_add_features=args.only_add_features,
    )

    # All done
    print("\nTraining complete.")
