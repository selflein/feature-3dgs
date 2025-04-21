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

import torch
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    gt_semantic_feature = cam_info.semantic_feature
    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )

    # image size will the same as feature map size
    elif args.resolution == 0:
        resolution = gt_semantic_feature.shape[2], gt_semantic_feature.shape[1]
    # customize resolution
    elif args.resolution == -2:
        resolution = 480, 320  # 800, 450

    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    if cam_info.mask is not None:
        resized_mask = torch.from_numpy(np.array(cam_info.mask.resize(resolution))).unsqueeze(0) / 255.0
    else:
        resized_mask = None

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        semantic_feature=gt_semantic_feature,
        data_device=args.data_device,
        mask=resized_mask,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


def camera_from_JSON(camera_entry, device):
    """Creates a Camera object from a JSON camera entry stored in `cameras.json`.

    This function inverts the camera_to_JSON process.

    Args:
        args: Arguments containing data_device and other settings.
        camera_entry: Dictionary containing camera parameters from JSON.
        resolution_scale: Scale factor for image resolution.

    Returns:
        A Camera object.
    """
    # Extract camera properties from JSON
    width = camera_entry["width"]
    height = camera_entry["height"]

    # Convert focal lengths to FoV angles
    fy = camera_entry["fy"]
    fx = camera_entry["fx"]
    FovY = 2 * np.arctan(height / (2 * fy))
    FovX = 2 * np.arctan(width / (2 * fx))

    # Reconstruct rotation and position
    pos = np.array(camera_entry["position"])
    rot = np.array(camera_entry["rotation"])

    # Calculate camera extrinsics (R, T)
    W2C = np.eye(4)
    W2C[:3, :3] = rot
    W2C[:3, 3] = pos

    Rt = np.linalg.inv(W2C)
    R = Rt[:3, :3].transpose()
    T = Rt[:3, 3]

    # Create a dummy image and mask since we don't have them from JSON
    dummy_image = torch.zeros((3, height, width), device=device)

    # Create a Camera object
    camera = Camera(
        colmap_id=camera_entry["id"],
        R=R,
        T=T,
        FoVx=FovX,
        FoVy=FovY,
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=camera_entry["img_name"],
        uid=camera_entry["id"],
        semantic_feature=None,  # No semantic feature provided in JSON
        data_device=device,
    )

    return camera
