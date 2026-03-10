#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Training script for point cloud segmentation with Sonata + SAM-3D + DINOv2.

Usage:
    python train_semantic_segmentation.py --data_root /path/to/data --batch_size 8 --epochs 100

For testing without real data:
    python train_semantic_segmentation.py --synthetic --epochs 10
"""

import os

# Skip sam3d_objects.init module which may not exist
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam3d_objects.semantic.model import PointCloudSegmentationModel
from sam3d_objects.semantic.loss import SegmentationLoss, get_loss_fn
from sam3d_objects.semantic.dataset import (
    PointCloudSegmentationDataset,
    SyntheticDataset,
    collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train point cloud segmentation model")
    
    # Data
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points per sample")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Model
    parser.add_argument("--sonata_model", type=str, default="sonata", help="Sonata model variant")
    parser.add_argument("--sam3d_config", type=str, default="checkpoints/hf/pipeline.yaml", help="SAM-3D config path")
    parser.add_argument("--fusion_dim", type=int, default=512, help="Fusion feature dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_attn_layers", type=int, default=4, help="Number of cross-attention layers")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze backbone weights")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Loss
    parser.add_argument("--loss_type", type=str, default="bce_dice", help="Loss type")
    parser.add_argument("--bce_weight", type=float, default=1.0, help="BCE loss weight")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Dice loss weight")
    parser.add_argument("--pos_weight", type=float, default=None, help="Positive class weight")
    
    # Misc
    parser.add_argument("--output_dir", type=str, default="outputs/semantic_segmentation", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval (epochs)")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval (epochs)")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU score."""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute segmentation metrics."""
    pred_binary = (pred > threshold).float()
    
    # True positives, false positives, false negatives
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    tn = ((1 - pred_binary) * (1 - target)).sum()
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    
    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
    }


class Trainer:
    """Trainer class for point cloud segmentation."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.global_step = 0
        self.best_iou = 0.0
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        # Build model
        self._build_model()
        
        # Build data loaders
        self._build_dataloaders()
        
        # Build optimizer and scheduler
        self._build_optimizer()
        
        # Build loss function
        self.loss_fn = get_loss_fn(
            loss_type=args.loss_type,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            pos_weight=args.pos_weight,
        )
        
        # AMP
        self.scaler = GradScaler() if args.amp else None
        
    def _build_model(self):
        """Build the segmentation model."""
        print("Building model...")
        
        self.model = PointCloudSegmentationModel(
            sonata_model=self.args.sonata_model,
            sam3d_config_path=self.args.sam3d_config,
            fusion_dim=self.args.fusion_dim,
            num_heads=self.args.num_heads,
            num_attn_layers=self.args.num_attn_layers,
            num_classes=1,
            freeze_backbone=self.args.freeze_backbone,
            device=str(self.device),
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def _build_dataloaders(self):
        """Build data loaders."""
        print("Building data loaders...")
        
        if self.args.synthetic:
            # Use synthetic data
            train_dataset = SyntheticDataset(num_samples=100, num_points=self.args.num_points)
            val_dataset = SyntheticDataset(num_samples=20, num_points=self.args.num_points)
        else:
            # Use real data
            train_dataset = PointCloudSegmentationDataset(
                data_root=self.args.data_root,
                split="train",
                num_points=self.args.num_points,
            )
            val_dataset = PointCloudSegmentationDataset(
                data_root=self.args.data_root,
                split="val",
                num_points=self.args.num_points,
            )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    def _build_optimizer(self):
        """Build optimizer and scheduler."""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        
        if self.args.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.lr * 0.01,
            )
        elif self.args.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            self.scheduler = None
            
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Process each sample in batch (due to variable point cloud sizes)
            batch_loss = 0.0
            batch_iou = 0.0
            batch_size = len(batch["points"])
            
            for i in range(batch_size):
                # Get single sample
                points = batch["points"][i].unsqueeze(0).to(self.device)
                colors = batch["colors"][i].unsqueeze(0).to(self.device)
                labels = batch["labels"][i].to(self.device)
                image = batch["image"][i:i+1].to(self.device)
                mask = batch["mask"][i:i+1].to(self.device)
                
                # Forward pass
                if self.args.amp:
                    with autocast():
                        logits = self.model(points, colors, image, mask)
                        pred = torch.sigmoid(logits.squeeze(-1))
                        loss = self.loss_fn(pred, labels)
                else:
                    logits = self.model(points, colors, image, mask)
                    pred = torch.sigmoid(logits.squeeze(-1))
                    loss = self.loss_fn(pred, labels)
                
                batch_loss += loss.item()
                batch_iou += compute_iou(pred, labels)
            
            # Average over batch
            loss = batch_loss / batch_size
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.args.amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
                self.optimizer.step()
            
            total_loss += batch_loss / batch_size
            total_iou += batch_iou / batch_size
            num_batches += 1
            
            # Logging
            if batch_idx % self.args.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{batch_loss / batch_size:.4f}",
                    "iou": f"{batch_iou / batch_size:.4f}",
                })
                
                self.writer.add_scalar("train/loss", batch_loss / batch_size, self.global_step)
                self.writer.add_scalar("train/iou", batch_iou / batch_size, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            
            self.global_step += 1
        
        return {
            "loss": total_loss / num_batches,
            "iou": total_iou / num_batches,
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            batch_loss = 0.0
            batch_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}
            batch_size = len(batch["points"])
            
            for i in range(batch_size):
                points = batch["points"][i].unsqueeze(0).to(self.device)
                colors = batch["colors"][i].unsqueeze(0).to(self.device)
                labels = batch["labels"][i].to(self.device)
                image = batch["image"][i:i+1].to(self.device)
                mask = batch["mask"][i:i+1].to(self.device)
                
                if self.args.amp:
                    with autocast():
                        logits = self.model(points, colors, image, mask)
                        pred = torch.sigmoid(logits.squeeze(-1))
                        loss = self.loss_fn(pred, labels)
                else:
                    logits = self.model(points, colors, image, mask)
                    pred = torch.sigmoid(logits.squeeze(-1))
                    loss = self.loss_fn(pred, labels)
                
                batch_loss += loss.item()
                metrics = compute_metrics(pred, labels)
                for k, v in metrics.items():
                    batch_metrics[k] += v
            
            total_loss += batch_loss / batch_size
            for k in total_metrics:
                total_metrics[k] += batch_metrics[k] / batch_size
            num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics["loss"] = total_loss / num_batches
        
        # Logging
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(self.args),
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        
        # Save periodic
        if epoch % self.args.save_interval == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch}.pth")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"New best model saved! IoU: {metrics['iou']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        start_epoch = 0
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)
        
        for epoch in range(start_epoch, self.args.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Validate
            if epoch % self.args.val_interval == 0 or epoch == self.args.epochs - 1:
                val_metrics = self.validate(epoch)
                print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}")
                
                # Check if best
                is_best = val_metrics["iou"] > self.best_iou
                if is_best:
                    self.best_iou = val_metrics["iou"]
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
        
        print(f"Training complete! Best IoU: {self.best_iou:.4f}")
        self.writer.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 50)
    print("Point Cloud Segmentation Training")
    print("=" * 50)
    print(f"Arguments: {args}")
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
