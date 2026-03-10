# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Dataset for point cloud segmentation with images and 2D masks.

Supports standard point cloud segmentation datasets like ScanNet, S3DIS, etc.
"""

import os
import json
import random
from typing import Optional, List, Dict, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class PointCloudSegmentationDataset(Dataset):
    """
    Dataset for point cloud segmentation.
    
    Expected data format:
    - point_cloud.npy: (N, 6) array with xyz coordinates and rgb colors
    - image.png: RGB image
    - mask.png: 2D object mask
    - label.npy: (N,) binary labels for point cloud
    - intrinsics.json: Camera intrinsics (optional)
    
    Or provide custom load functions.
    
    Args:
        data_root: Root directory of the dataset
        split: Dataset split (train/val/test)
        num_points: Number of points to sample (None for all)
        color_jitter: Apply color jittering augmentation
        random_rotate: Apply random rotation augmentation
        random_scale: Apply random scale augmentation
        random_flip: Apply random flip augmentation
        transform: Custom transform function
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_points: Optional[int] = None,
        color_jitter: bool = True,
        random_rotate: bool = True,
        random_scale: bool = True,
        random_flip: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.color_jitter = color_jitter
        self.random_rotate = random_rotate
        self.random_scale = random_scale
        self.random_flip = random_flip
        self.transform = transform
        
        # Load scene list
        self.scenes = self._load_scene_list()
        
        # Augmentation parameters
        self.scale_range = (0.8, 1.2)
        self.flip_prob = 0.5
        self.rotate_range = (-np.pi, np.pi)
        
    def _load_scene_list(self) -> List[str]:
        """Load list of scene directories."""
        split_file = os.path.join(self.data_root, f"{self.split}.txt")
        
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                scenes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Assume all subdirectories are scenes
            scenes = [
                d for d in os.listdir(self.data_root)
                if os.path.isdir(os.path.join(self.data_root, d))
                and not d.startswith(".")
            ]
        
        return scenes
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_name = self.scenes[idx]
        scene_dir = os.path.join(self.data_root, scene_name)
        
        # Load point cloud
        point_cloud = self._load_point_cloud(scene_dir)
        points = point_cloud[:, :3]  # xyz
        colors = point_cloud[:, 3:6] / 255.0  # rgb, normalize to [0, 1]
        
        # Load labels
        labels = self._load_labels(scene_dir, len(points))
        
        # Load image and mask
        image, mask = self._load_image_and_mask(scene_dir)
        
        # Load camera intrinsics (optional)
        intrinsics = self._load_intrinsics(scene_dir)
        
        # Sample points if specified
        if self.num_points is not None and len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            colors = colors[indices]
            labels = labels[indices]
        
        # Apply augmentations
        if self.split == "train":
            points, colors = self._apply_augmentations(points, colors)
        
        # Convert to tensors
        data = {
            "points": torch.from_numpy(points).float(),
            "colors": torch.from_numpy(colors).float(),
            "labels": torch.from_numpy(labels).float(),
            "image": torch.from_numpy(image).float().permute(2, 0, 1) / 255.0,  # (3, H, W)
            "mask": torch.from_numpy(mask).float().unsqueeze(0),  # (1, H, W)
            "scene_name": scene_name,
        }
        
        if intrinsics is not None:
            data["intrinsics"] = torch.from_numpy(intrinsics).float()
        
        return data
    
    def _load_point_cloud(self, scene_dir: str) -> np.ndarray:
        """Load point cloud from file."""
        # Try different formats
        for ext in [".npy", ".bin", ".ply"]:
            pc_path = os.path.join(scene_dir, f"point_cloud{ext}")
            if os.path.exists(pc_path):
                if ext == ".npy":
                    return np.load(pc_path)
                elif ext == ".bin":
                    return np.fromfile(pc_path, dtype=np.float32).reshape(-1, 6)
                elif ext == ".ply":
                    return self._load_ply(pc_path)
        
        raise FileNotFoundError(f"No point cloud found in {scene_dir}")
    
    def _load_ply(self, path: str) -> np.ndarray:
        """Load PLY file."""
        from plyfile import PlyData
        plydata = PlyData.read(path)
        points = np.vstack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z'],
            plydata['vertex']['red'],
            plydata['vertex']['green'],
            plydata['vertex']['blue'],
        ]).T
        return points
    
    def _load_labels(self, scene_dir: str, num_points: int) -> np.ndarray:
        """Load point labels."""
        label_path = os.path.join(scene_dir, "label.npy")
        
        if os.path.exists(label_path):
            return np.load(label_path)
        else:
            # Return zeros if no labels (for inference)
            return np.zeros(num_points, dtype=np.float32)
    
    def _load_image_and_mask(self, scene_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load RGB image and 2D mask."""
        # Load image
        image_path = os.path.join(scene_dir, "image.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(scene_dir, "rgb.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(scene_dir, "color.png")
        
        if os.path.exists(image_path):
            image = np.array(Image.open(image_path).convert("RGB"))
        else:
            raise FileNotFoundError(f"No image found in {scene_dir}")
        
        # Load mask
        mask_path = os.path.join(scene_dir, "mask.png")
        if not os.path.exists(mask_path):
            # Try to find any mask file
            for f in os.listdir(scene_dir):
                if "mask" in f.lower() and f.endswith(".png"):
                    mask_path = os.path.join(scene_dir, f)
                    break
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 127).astype(np.float32)
        else:
            # Create dummy mask (all ones) if no mask found
            mask = np.ones(image.shape[:2], dtype=np.float32)
        
        return image, mask
    
    def _load_intrinsics(self, scene_dir: str) -> Optional[np.ndarray]:
        """Load camera intrinsics."""
        intrinsics_path = os.path.join(scene_dir, "intrinsics.json")
        
        if os.path.exists(intrinsics_path):
            with open(intrinsics_path, "r") as f:
                intrinsics_dict = json.load(f)
            
            # Create 3x3 intrinsics matrix
            intrinsics = np.array([
                [intrinsics_dict.get("fx", 500), 0, intrinsics_dict.get("cx", 320)],
                [0, intrinsics_dict.get("fy", 500), intrinsics_dict.get("cy", 240)],
                [0, 0, 1],
            ])
            return intrinsics
        
        return None
    
    def _apply_augmentations(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentations."""
        # Random rotation around Z axis
        if self.random_rotate:
            angle = np.random.uniform(*self.rotate_range)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1],
            ])
            points = points @ rotation_matrix.T
        
        # Random scale
        if self.random_scale:
            scale = np.random.uniform(*self.scale_range)
            points = points * scale
        
        # Random flip
        if self.random_flip:
            if np.random.random() < self.flip_prob:
                points[:, 0] = -points[:, 0]  # Flip X
        
        # Color jitter
        if self.color_jitter:
            # Add random brightness and contrast
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            colors = np.clip(colors * brightness * contrast, 0, 1)
        
        return points, colors


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for DataLoader.
    
    Handles variable-size point clouds by keeping them as lists.
    """
    # Stack fixed-size tensors
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    
    # Keep variable-size point clouds as lists
    points_list = [item["points"] for item in batch]
    colors_list = [item["colors"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    
    # Compute offsets for batch processing
    offsets = torch.cumsum(torch.tensor([len(p) for p in points_list]), dim=0)
    
    # Scene names
    scene_names = [item["scene_name"] for item in batch]
    
    # Optional intrinsics
    if "intrinsics" in batch[0]:
        intrinsics = torch.stack([item["intrinsics"] for item in batch])
    else:
        intrinsics = None
    
    return {
        "points": points_list,
        "colors": colors_list,
        "labels": labels_list,
        "image": images,
        "mask": masks,
        "offsets": offsets,
        "scene_names": scene_names,
        "intrinsics": intrinsics,
    }


def create_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create DataLoader for point cloud segmentation.
    
    Args:
        data_root: Root directory of the dataset
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of workers
        **dataset_kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader
    """
    dataset = PointCloudSegmentationDataset(
        data_root=data_root,
        split=split,
        **dataset_kwargs,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    
    return dataloader


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing without real data.
    
    Generates random point clouds, images, and masks.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_points: int = 1024,
        image_size: int = 224,
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.image_size = image_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random point cloud
        points = torch.randn(self.num_points, 3) * 0.5  # Centered at origin
        colors = torch.rand(self.num_points, 3)  # Random colors
        
        # Random image
        image = torch.rand(3, self.image_size, self.image_size)
        
        # Random mask (ellipse-like region)
        mask = torch.zeros(1, self.image_size, self.image_size)
        center = torch.randint(self.image_size // 4, 3 * self.image_size // 4, (2,))
        for i in range(self.image_size):
            for j in range(self.image_size):
                if ((i - center[0]) ** 2 + (j - center[1]) ** 2) < (self.image_size // 4) ** 2:
                    mask[0, i, j] = 1.0
        
        # Random labels (points near center are positive)
        center_3d = torch.randn(3) * 0.1
        distances = (points - center_3d).norm(dim=1)
        labels = (distances < 0.3).float()
        
        return {
            "points": points,
            "colors": colors,
            "labels": labels,
            "image": image,
            "mask": mask,
            "scene_name": f"synthetic_{idx:04d}",
        }
