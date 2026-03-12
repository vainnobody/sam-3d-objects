from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from dataset.fusion_utils import PointCloudToImageMapper
from model import GaussianModel, render
from scene import Scene
from utils.system_utils import searchForMaxIteration


@dataclass
class GaussianSceneBundle:
    scene: Scene
    gaussians: GaussianModel
    train_views: Any
    camera_infos: list[Any]


def load_scene_and_gaussians(config) -> GaussianSceneBundle:
    scene = Scene(config.scene)
    gaussians = GaussianModel(config.model.sh_degree)
    loaded_iter = config.model.load_iteration
    if loaded_iter == -1:
        loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, 'point_cloud'))
    ply_path = os.path.join(
        config.model.model_dir,
        'point_cloud',
        f'iteration_{loaded_iter}',
        'point_cloud.ply',
    )
    gaussians.load_ply(ply_path)
    train_views = scene.getTrainCameras()
    return GaussianSceneBundle(
        scene=scene,
        gaussians=gaussians,
        train_views=train_views,
        camera_infos=train_views.camera_info,
    )


def build_intrinsics(camera, image_height: int, image_width: int) -> np.ndarray:
    fx = image_width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    fy = image_height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    intrinsics = np.array(
        [
            [fx, 0.0, image_width / 2.0],
            [0.0, fy, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics


def extract_gaussian_attributes(gaussians: GaussianModel) -> dict[str, torch.Tensor]:
    rgb = gaussians._features_dc[:, 0, :]
    return {
        'xyz': gaussians.get_xyz.detach(),
        'rgb': rgb.detach(),
        'opacity': gaussians.get_opacity.detach(),
        'scaling': gaussians._scaling.detach(),
        'rotation': gaussians._rotation.detach(),
    }


def build_attribute_tensor(attr: dict[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
    parts = [
        attr['rgb'][indices],
        attr['opacity'][indices],
        attr['scaling'][indices],
        attr['rotation'][indices],
    ]
    return torch.cat(parts, dim=-1)


def render_depth_map(view, gaussians: GaussianModel, pipeline_config, image_hw: tuple[int, int]) -> np.ndarray:
    view.cuda()
    bg = torch.zeros(3, dtype=torch.float32, device='cuda')
    render_pkg = render(
        view,
        gaussians,
        pipeline_config,
        bg,
        override_shape=(image_hw[1], image_hw[0]),
    )
    return render_pkg['depth'].detach().cpu().numpy()[0]


def compute_view_mapping(view, camera_info, xyz: torch.Tensor, depth: np.ndarray, image_hw: tuple[int, int], visibility_threshold: float, cut_boundary: int) -> tuple[np.ndarray, np.ndarray]:
    mapper = PointCloudToImageMapper(
        image_dim=[image_hw[1], image_hw[0]],
        visibility_threshold=visibility_threshold,
        cut_bound=cut_boundary,
        intrinsics=camera_info.intrinsics,
    )
    mapping, weight = mapper.compute_mapping(
        view.world_view_transform.detach().cpu().numpy(),
        xyz.detach().cpu().numpy(),
        depth,
    )
    return mapping, weight


def discover_mask_files(mask_root: Path, image_name: str) -> list[Path]:
    candidates = []
    for ext in ('.png', '.jpg', '.jpeg', '.webp'):
        direct = mask_root / f'{image_name}{ext}'
        if direct.exists():
            candidates.append(direct)
    nested_root = mask_root / image_name
    if nested_root.is_dir():
        for item in sorted(nested_root.iterdir()):
            if item.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                candidates.append(item)
    unique = []
    seen = set()
    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def load_mask(mask_path: Path, image_hw: tuple[int, int]) -> torch.Tensor:
    mask = Image.open(mask_path).convert('L').resize((image_hw[1], image_hw[0]))
    mask_np = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(mask_np).unsqueeze(0)


def _load_optional_array(path: Path | None, key: str, expected: int) -> torch.Tensor:
    if path is None or not path.exists():
        if key == 'surface_distance':
            return torch.ones(expected, 1, dtype=torch.float32)
        return torch.zeros(expected, 1, dtype=torch.float32)
    data = np.load(path)
    if key not in data:
        if key == 'surface_distance':
            return torch.ones(expected, 1, dtype=torch.float32)
        return torch.zeros(expected, 1, dtype=torch.float32)
    array = torch.from_numpy(np.asarray(data[key], dtype=np.float32))
    if array.ndim == 1:
        array = array.unsqueeze(-1)
    if array.shape[0] == expected:
        return array
    raise ValueError(f'{path}::{key} has length {array.shape[0]} but expected {expected}')


def discover_prior_file(prior_root: Path | None, image_name: str, object_name: str) -> Path | None:
    if prior_root is None:
        return None
    patterns = [
        prior_root / image_name / f'{object_name}.npz',
        prior_root / f'{image_name}_{object_name}.npz',
        prior_root / f'{image_name}.npz',
    ]
    for item in patterns:
        if item.exists():
            return item
    return None


def discover_label_file(label_root: Path | None, image_name: str, object_name: str) -> Path | None:
    if label_root is None:
        return None
    patterns = [
        label_root / image_name / f'{object_name}.npy',
        label_root / f'{image_name}_{object_name}.npy',
        label_root / f'{image_name}.npy',
    ]
    for item in patterns:
        if item.exists():
            return item
    return None


def load_labels(label_path: Path | None, expected: int) -> torch.Tensor | None:
    if label_path is None or not label_path.exists():
        return None
    labels = torch.from_numpy(np.asarray(np.load(label_path), dtype=np.float32))
    if labels.ndim > 1:
        labels = labels.squeeze(-1)
    if labels.shape[0] != expected:
        raise ValueError(f'{label_path} has {labels.shape[0]} labels, expected {expected}')
    return labels
