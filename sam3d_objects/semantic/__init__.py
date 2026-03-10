# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Semantic segmentation module for point clouds with Sonata + SAM-3D + DINOv2.
"""

from .model import (
    PointCloudSegmentationModel,
    CrossAttentionFusion,
    SegmentationDecoder,
)
from .loss import (
    SegmentationLoss,
    BoundaryLoss,
    LovaszLoss,
    get_loss_fn,
)
from .dataset import (
    PointCloudSegmentationDataset,
    SyntheticDataset,
    collate_fn,
    create_dataloader,
)

__all__ = [
    # Model
    "PointCloudSegmentationModel",
    "CrossAttentionFusion",
    "SegmentationDecoder",
    # Loss
    "SegmentationLoss",
    "BoundaryLoss",
    "LovaszLoss",
    "get_loss_fn",
    # Dataset
    "PointCloudSegmentationDataset",
    "SyntheticDataset",
    "collate_fn",
    "create_dataloader",
]