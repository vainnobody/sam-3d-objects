# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Loss functions for point cloud segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SegmentationLoss(nn.Module):
    """
    Combined loss for point cloud segmentation.
    
    Combines BCE Loss and Dice Loss for handling class imbalance.
    
    Args:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        focal_weight: Weight for Focal loss (optional)
        focal_gamma: Focal loss gamma parameter
        pos_weight: Positive class weight for BCE
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        
        # BCE loss with optional pos_weight for class imbalance
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        else:
            self.pos_weight = None
    
    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: (B, N) predicted probabilities
            target: (B, N) target labels
            smooth: Smoothing factor
        
        Returns:
            dice_loss: scalar
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice
    
    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: (B, N) predicted probabilities
            target: (B, N) target labels
        
        Returns:
            focal_loss: scalar
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.focal_gamma) * bce
        return focal.mean()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: (B, N) predicted probabilities (after sigmoid)
            target: (B, N) target labels (0 or 1)
        
        Returns:
            total_loss: scalar
        """
        total_loss = 0.0
        
        # BCE Loss
        if self.bce_weight > 0:
            bce = F.binary_cross_entropy(
                pred, target, 
                pos_weight=self.pos_weight,
                reduction='mean'
            )
            total_loss = total_loss + self.bce_weight * bce
        
        # Dice Loss
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            total_loss = total_loss + self.dice_weight * dice
        
        # Focal Loss
        if self.focal_weight > 0:
            focal = self.focal_loss(pred, target)
            total_loss = total_loss + self.focal_weight * focal
        
        return total_loss


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for better segmentation boundaries.
    
    Encourages smooth boundaries by penalizing high gradients
    near boundary points.
    """
    
    def __init__(self, boundary_weight: float = 0.1):
        super().__init__()
        self.boundary_weight = boundary_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute boundary loss.
        
        Args:
            pred: (B, N) predicted probabilities
            target: (B, N) target labels
            boundary_mask: (B, N) optional mask for boundary points
        
        Returns:
            boundary_loss: scalar
        """
        # Compute gradient
        pred_grad = pred[:, 1:] - pred[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        
        # L1 loss on gradients
        grad_loss = F.l1_loss(pred_grad.abs(), target_grad.abs())
        
        if boundary_mask is not None:
            # Apply boundary mask
            boundary_pred = pred * boundary_mask
            boundary_target = target * boundary_mask
            boundary_bce = F.binary_cross_entropy(boundary_pred, boundary_target)
            return self.boundary_weight * (grad_loss + boundary_bce)
        
        return self.boundary_weight * grad_loss


class LovaszLoss(nn.Module):
    """
    Lovasz-Softmax loss for semantic segmentation.
    
    A surrogate loss for IoU optimization.
    """
    
    def __init__(self):
        super().__init__()
    
    def lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of the Lovasz extension w.r.t sorted errors.
        
        Args:
            gt_sorted: (P) sorted ground truth labels
        
        Returns:
            grad: (P) gradient
        """
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        
        return jaccard
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Lovasz loss for binary segmentation.
        
        Args:
            pred: (B, N) predicted probabilities
            target: (B, N) target labels
        
        Returns:
            lovasz_loss: scalar
        """
        B, N = pred.shape
        
        total_loss = 0.0
        for b in range(B):
            pred_b = pred[b]
            target_b = target[b]
            
            # Sort by prediction
            errors = (pred_b - target_b).abs()
            sorted_errors, indices = torch.sort(errors, descending=True)
            sorted_target = target_b[indices]
            
            # Compute gradient
            grad = self.lovasz_grad(sorted_target)
            
            # Compute loss
            loss = (sorted_errors * grad).sum()
            total_loss = total_loss + loss
        
        return total_loss / B


def get_loss_fn(
    loss_type: str = "bce_dice",
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: Optional[float] = None,
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: Type of loss function
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        pos_weight: Positive class weight
    
    Returns:
        loss_fn: Loss function module
    """
    if loss_type == "bce_dice":
        return SegmentationLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
        )
    elif loss_type == "bce":
        return SegmentationLoss(
            bce_weight=1.0,
            dice_weight=0.0,
            pos_weight=pos_weight,
        )
    elif loss_type == "dice":
        return SegmentationLoss(
            bce_weight=0.0,
            dice_weight=1.0,
        )
    elif loss_type == "focal":
        return SegmentationLoss(
            bce_weight=0.0,
            dice_weight=0.0,
            focal_weight=1.0,
        )
    elif loss_type == "lovasz":
        return LovaszLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
