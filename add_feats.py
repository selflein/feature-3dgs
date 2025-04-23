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

"""Adds semantic features to an existing 3DGS model."""

import json
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from arguments import ModelParams, OptimizationParams, PipelineParams
from einops import asnumpy
from gaussian_renderer import render
from PIL import Image as PILImage
from scene import GaussianModel
from scene.cameras import Camera
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from utils.camera_utils import camera_from_JSON
from utils.general_utils import PILtoTorch
from utils.image_utils import psnr
from utils.loss_utils import l1_loss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def extract_feats(img_paths: list[Path], output_dir: Path, device: str, checkpoint: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    in_out = ((img_path, output_dir / f"{img_path.stem}_fmap_CxHxW.pt") for img_path in img_paths)
    in_out = [(i, o) for i, o in in_out if not o.is_file()]

    if not in_out:
        return

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    for img_path, output_feat_path in tqdm(in_out, "Extracting SAM features for images."):
        image = np.array(PILImage.open(img_path)) # (1423, 1908, 3)
        predictor.set_image(image)
        image_embedding_tensor = torch.tensor(predictor.get_image_embedding().cpu().numpy()[0])

        img_h, img_w, _ = image.shape
        _, fea_h, fea_w = image_embedding_tensor.shape
        cropped_h = int(fea_w / img_w * img_h + 0.5)
        image_embedding_tensor_cropped = image_embedding_tensor[:, :cropped_h, :]
        torch.save(image_embedding_tensor_cropped, output_feat_path)

    del predictor, sam
    torch.cuda.empty_cache()


def prepare(output_dir: Path, views: list[Camera], gaussians: GaussianModel, pipe_config: PipelineParams, background: torch.Tensor, sam_checkpoint: str):
    """Render views from existing Gaussian splatting model and extract 2D SAM features."""
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for view in tqdm(views, "Rendering cameras."):
        img_name = f"{view.colmap_id}.png"
        img_output_path = img_dir / img_name
        mask_output_path = mask_dir / img_name

        if img_output_path.exists() and mask_output_path.exists():
            rgb = PILImage.open(img_output_path)
            mask = PILImage.open(mask_output_path)
        else:
            with torch.inference_mode():
                render_pkg = render(view, gaussians, pipe_config, background)

            rgb = (asnumpy(render_pkg['render'].permute(1, 2, 0)).clip(0, 1) * 255).astype(np.uint8)
            mask = (rgb != asnumpy(background)[None, None, :]).all(axis=-1)

            mask = PILImage.fromarray((mask* 255).astype(np.uint8))
            rgb = PILImage.fromarray(rgb)
            with open(mask_output_path, "wb") as f:
                mask.save(f)

            with open(img_output_path, "wb") as f:
                rgb.save(f)

        view.mask = PILtoTorch(mask)
        view.original_image = PILtoTorch(rgb)

    extract_feats(list(img_dir.glob("*.png")), output_dir / "sam_embeddings", background.device, sam_checkpoint)

    for view in tqdm(views, "Loading SAM features."):
        view.semantic_feature = torch.load(output_dir / "sam_embeddings" / f"{view.colmap_id}_fmap_CxHxW.pt")


def training(args, opt, pipe, saving_iterations, cameras_json_file, gs_path, output_dir, sam_checkpoint):
    with open(cameras_json_file, "r") as f:
        cameras = [camera_from_JSON(cam_data, torch.device("cpu")) for cam_data in json.load(f)]

    first_iter = 0
    tb_writer = prepare_output_and_logger(args, output_dir)
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(gs_path)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    prepare(output_dir, cameras, gaussians, pipe, background, sam_checkpoint)

    viewpoint_stack = cameras.copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()

    gaussians.training_setup(opt)
    gaussians.set_gaussian_opt_enabled(False)  # Only optimize the semantic features

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, max(saving_iterations)), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, max(saving_iterations)):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)

        feature_map, image = render_pkg["feature_map"], render_pkg["render"]

        # Loss
        mask = viewpoint_cam.mask.cuda(non_blocking=True)
        gt_image = viewpoint_cam.original_image.cuda(non_blocking=True)

        ignore_mask = mask[0] == 0
        image[:, ignore_mask] = gt_image[:, ignore_mask]

        Ll1 = l1_loss(image, gt_image)

        gt_feature_map = viewpoint_cam.semantic_feature.cuda(non_blocking=True)
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) 
        feature_ignore_mask = F.interpolate(mask.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
        feature_ignore_mask = feature_ignore_mask[0] == 0
        feature_map[:, feature_ignore_mask] = gt_feature_map[:, feature_ignore_mask]

        Ll1_feature = l1_loss(feature_map, gt_feature_map) 
        loss = Ll1_feature 

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            num_cams_to_render = 5
            cam_subsample_stride = len(cameras) // num_cams_to_render
            render_cams = cameras[::cam_subsample_stride]

            # Log and save
            training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, iter_start.elapsed_time(iter_end), saving_iterations, render, (pipe, background), gaussians, render_cams) 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                gaussians.save_ply(output_dir / f"point_cloud_iter_{iteration}.ply")
  
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


def prepare_output_and_logger(args, output_dir):    
    args.model_path = Path(output_dir) / "logs"
    args.model_path.mkdir(parents=True, exist_ok=True)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, elapsed, testing_iterations, renderFunc, renderArgs, gaussians, train_cams):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'train', 'cameras' : train_cams},)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(0, 1_000, 100)))
    parser.add_argument("--gs_path", type=Path, required=True)
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sam_checkpoint", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Optimizing " + args.model_path)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.save_iterations,
        gs_path=args.gs_path,
        cameras_json_file=args.cameras_json,
        output_dir=args.output_dir,
        sam_checkpoint=args.sam_checkpoint,
    )
    print("\nTraining complete.")
