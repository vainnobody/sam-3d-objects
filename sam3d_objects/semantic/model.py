# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Point Cloud Segmentation Model with Sonata + SAM-3D + DINOv2

This model performs point cloud segmentation by fusing three types of features:
1. Sonata point cloud features (geometric)
2. SAM-3D Stage 1 geometric priors (voxel, shape latent)
3. DINOv2 visual features (semantic)

The fusion is done via Cross-Attention where Sonata features serve as queries
and DINOv2 features serve as keys/values.
"""

import os
import sys
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# Add notebook path for stage1_inference
NOTEBOOK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "notebook")
if NOTEBOOK_PATH not in sys.path:
    sys.path.insert(0, NOTEBOOK_PATH)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention fusion module for point cloud and image features.
    
    Point cloud features serve as queries.
    Image (DINOv2) features serve as keys and values.
    Geometric features can be added as auxiliary conditioning.
    """
    
    def __init__(
        self,
        point_dim: int = 512,
        image_dim: int = 768,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_geometric_feat: bool = True,
        geometric_dim: int = 8,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_layers = num_layers
        self.use_geometric_feat = use_geometric_feat
        
        # Project point cloud features to fusion dimension
        self.point_proj = nn.Linear(point_dim, fusion_dim)
        
        # Project image features to fusion dimension
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        
        # Project geometric features (optional)
        if use_geometric_feat:
            self.geo_proj = nn.Linear(geometric_dim, fusion_dim)
        
        # Cross-Attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms and FFN
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(fusion_dim) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(fusion_dim) for _ in range(num_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 4, fusion_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        point_feat: torch.Tensor,
        image_feat: torch.Tensor,
        geometric_feat: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            point_feat: (B, N, point_dim) - Sonata point cloud features
            image_feat: (B, L, image_dim) - DINOv2 image features (L = H/14 * W/14)
            geometric_feat: (B, N, geometric_dim) - SAM-3D geometric features (optional)
            point_mask: (B, N) - Valid point mask (optional)
        
        Returns:
            fused_feat: (B, N, fusion_dim) - Fused features
        """
        B, N, _ = point_feat.shape
        
        # Project features
        point_proj = self.point_proj(point_feat)  # (B, N, fusion_dim)
        image_proj = self.image_proj(image_feat)  # (B, L, fusion_dim)
        
        # Add geometric features
        if self.use_geometric_feat and geometric_feat is not None:
            geo_proj = self.geo_proj(geometric_feat)  # (B, N, fusion_dim)
            point_proj = point_proj + geo_proj
        
        # Cross-Attention: point as query, image as key/value
        h = point_proj
        for i in range(self.num_layers):
            # Self-norm before attention
            h_norm = self.layer_norms1[i](h)
            
            # Cross-attention
            attn_out, _ = self.cross_attn_layers[i](
                query=h_norm,
                key=image_proj,
                value=image_proj,
                key_padding_mask=None,
            )
            
            # Residual
            h = h + attn_out
            
            # FFN
            h_norm = self.layer_norms2[i](h)
            h = h + self.ffn_layers[i](h_norm)
        
        return h


class SegmentationDecoder(nn.Module):
    """
    Simple MLP decoder for point cloud segmentation.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: list = [256, 128],
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_dim) fused features
        
        Returns:
            logits: (B, N, num_classes)
        """
        return self.decoder(x)


class PointCloudSegmentationModel(nn.Module):
    """
    Point Cloud Segmentation Model
    
    Fuses Sonata point cloud features with SAM-3D Stage 1 geometric priors
    and DINOv2 visual features for point cloud segmentation.
    
    Args:
        sonata_model: Sonata model name ("sonata", "sonata_small")
        sam3d_config_path: Path to SAM-3D pipeline.yaml config
        fusion_dim: Fusion feature dimension
        num_heads: Number of attention heads
        num_attn_layers: Number of cross-attention layers
        num_classes: Number of output classes (1 for binary segmentation)
        freeze_backbone: Whether to freeze Sonata and SAM-3D
        use_geometric_feat: Whether to use SAM-3D geometric features
    """
    
    def __init__(
        self,
        sonata_model: str = "sonata",
        sam3d_config_path: str = "checkpoints/hf/pipeline.yaml",
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_attn_layers: int = 4,
        num_classes: int = 1,
        freeze_backbone: bool = True,
        use_geometric_feat: bool = True,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.freeze_backbone = freeze_backbone
        self.use_geometric_feat = use_geometric_feat
        self.fusion_dim = fusion_dim
        
        # 1. Load Sonata model
        logger.info(f"Loading Sonata model: {sonata_model}")
        import sonata
        self.sonata = sonata.load(sonata_model).to(self.device)
        self.sonata.eval()
        
        # Get Sonata output dimension
        self.sonata_dim = 512  # Sonata outputs 512-dim features
        
        # 2. Load SAM-3D Stage 1 model
        logger.info(f"Loading SAM-3D Stage 1 from: {sam3d_config_path}")
        from stage1_inference import Stage1OnlyInference
        self.sam3d_stage1 = Stage1OnlyInference(sam3d_config_path)
        self.sam3d_stage1.eval()
        
        # DINOv2 dimension (dinov2_vitb14)
        self.dino_dim = 768
        
        # Geometric feature dimension
        self.geo_dim = 8  # shape latent channels
        
        # Freeze backbones
        if freeze_backbone:
            logger.info("Freezing Sonata and SAM-3D backbones...")
            for param in self.sonata.parameters():
                param.requires_grad = False
            for param in self.sam3d_stage1.parameters():
                param.requires_grad = False
        
        # 3. Feature fusion module
        self.fusion = CrossAttentionFusion(
            point_dim=self.sonata_dim,
            image_dim=self.dino_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            num_layers=num_attn_layers,
            dropout=dropout,
            use_geometric_feat=use_geometric_feat,
            geometric_dim=self.geo_dim,
        )
        
        # 4. Segmentation decoder
        self.decoder = SegmentationDecoder(
            input_dim=fusion_dim,
            hidden_dims=[256, 128],
            num_classes=num_classes,
            dropout=dropout,
        )
        
        # 5. Sonata transform
        import sonata.transform as transform
        self.sonata_transform = transform.default()
        
    def extract_sonata_features(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract point cloud features using Sonata.
        
        Args:
            points: (B, N, 3) point coordinates
            colors: (B, N, 3) point colors [0, 1] range
        
        Returns:
            features: (B, N, 512) point features
        """
        B, N, _ = points.shape
        
        # Prepare point dict for Sonata
        # Sonata expects coordinates and features (colors)
        point_dict = {
            "coord": points[0].cpu().numpy(),  # (N, 3)
            "color": (colors[0] * 255).cpu().numpy().astype("uint8"),  # (N, 3), [0, 255]
        }
        
        # Apply Sonata transform
        point_dict = self.sonata_transform(point_dict)
        
        # Move to device
        for key in point_dict.keys():
            if isinstance(point_dict[key], torch.Tensor):
                point_dict[key] = point_dict[key].to(self.device).unsqueeze(0)  # Add batch dim
        
        # Extract features
        with torch.no_grad():
            point_out = self.sonata(point_dict)
        
        # Get features - need to upsample back to original resolution
        # Sonata downsamples during GridSample, we need to map back
        feat = point_out.feat  # (1, M, 512) where M < N due to GridSample
        
        # Simple nearest neighbor upsampling for now
        # In practice, you might want to use the inverse mapping from Sonata
        if feat.shape[1] != N:
            # Get original coordinates
            original_coord = point_dict["coord"]  # (1, N, 3)
            downsampled_coord = point_out.coord  # (1, M, 3)
            
            # Nearest neighbor interpolation
            # Expand to compute distances
            dist = torch.cdist(original_coord[0], downsampled_coord[0])  # (N, M)
            nearest_idx = dist.argmin(dim=1)  # (N,)
            
            # Gather features
            feat = feat[0, nearest_idx]  # (N, 512)
            feat = feat.unsqueeze(0)  # (1, N, 512)
        
        return feat
    
    def extract_sam3d_features(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract geometric and DINOv2 features using SAM-3D Stage 1.
        
        Args:
            image: (B, 3, H, W) RGB image [0, 1] range
            mask: (B, 1, H, W) binary mask
        
        Returns:
            dict with:
                - dino_features: (B, L, 768) DINOv2 features
                - voxel: (M, 3) sparse voxel coordinates
                - shape: (B, 8, 16, 16, 16) shape latent
        """
        B = image.shape[0]
        
        # Convert to numpy format expected by SAM-3D
        # image: (B, 3, H, W) -> (H, W, 3) uint8
        # mask: (B, 1, H, W) -> (H, W) bool
        image_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        mask_np = mask[0, 0].cpu().numpy().astype(bool)
        
        # Run SAM-3D Stage 1
        with torch.no_grad():
            output = self.sam3d_stage1.run(
                image=image_np,
                mask=mask_np,
                seed=42,
                steps=4,
                use_distillation=True,
            )
        
        # Extract DINOv2 features from condition embedder
        # We need to access the internal DINO features
        # The embedder output is already computed during SAM-3D inference
        
        # Get DINOv2 features from the model
        # SAM-3D uses DINOv2 ViT-B/14 with 768-dim features
        # The feature map shape is (B, num_patches + 1, 768)
        # We need spatial features without CLS token
        
        # For now, we'll re-extract DINO features directly
        from sam3d_objects.model.backbone.dit.embedder.dino import Dino
        dino_embedder = Dino(
            input_size=224,
            repo_or_dir="facebookresearch/dinov2",
            dino_model="dinov2_vitb14",
            source="github",
            freeze_backbone=True,
        ).to(self.device)
        dino_embedder.eval()
        
        # Preprocess image for DINO
        # Resize to 224x224 and normalize
        image_dino = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        image_dino = (image_dino - mean) / std
        
        with torch.no_grad():
            dino_tokens = dino_embedder(image_dino)  # (B, num_patches + 1, 768)
        
        # Remove CLS token, keep spatial patches
        dino_features = dino_tokens[:, 1:, :]  # (B, L, 768), L = 16*16 = 256 for 224x224 input
        
        return {
            "dino_features": dino_features,
            "voxel": output["voxel"],  # (M, 3)
            "shape": output["shape"],  # (B, 8, 16, 16, 16)
        }
    
    def interpolate_shape_to_points(
        self,
        shape_latent: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate shape latent to point coordinates.
        
        Args:
            shape_latent: (B, 8, 16, 16, 16) shape latent
            points: (B, N, 3) point coordinates, normalized to [-0.5, 0.5]
        
        Returns:
            point_geo_feat: (B, N, 8) interpolated features
        """
        B, C, D, H, W = shape_latent.shape
        _, N, _ = points.shape
        
        # Normalize points to [-1, 1] for grid_sample
        # points are in [-0.5, 0.5], so multiply by 2
        points_norm = points * 2.0  # (B, N, 3) in [-1, 1]
        
        # Reorder coordinates for grid_sample (x, y, z) -> (z, y, x) for (D, H, W)
        points_grid = points_norm[..., [2, 1, 0]]  # (B, N, 3)
        
        # Reshape for grid_sample
        grid = points_grid.view(B, 1, 1, N, 3)  # (B, 1, 1, N, 3)
        
        # Trilinear interpolation
        point_geo_feat = F.grid_sample(
            shape_latent,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )  # (B, C, 1, 1, N)
        
        point_geo_feat = point_geo_feat.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        
        return point_geo_feat
    
    def forward(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for point cloud segmentation.
        
        Args:
            points: (B, N, 3) point coordinates
            colors: (B, N, 3) point colors [0, 1]
            image: (B, 3, H, W) RGB image [0, 1]
            mask: (B, 1, H, W) 2D object mask
        
        Returns:
            logits: (B, N, num_classes) segmentation logits
        """
        B, N, _ = points.shape
        
        # 1. Extract Sonata point cloud features
        point_feat = self.extract_sonata_features(points, colors)  # (B, N, 512)
        
        # 2. Extract SAM-3D features (DINOv2 + geometric)
        sam3d_output = self.extract_sam3d_features(image, mask)
        dino_feat = sam3d_output["dino_features"]  # (B, L, 768)
        shape_latent = sam3d_output["shape"]  # (B, 8, 16, 16, 16)
        
        # 3. Interpolate shape latent to point coordinates
        if self.use_geometric_feat:
            geo_feat = self.interpolate_shape_to_points(shape_latent, points)  # (B, N, 8)
        else:
            geo_feat = None
        
        # 4. Fuse features via Cross-Attention
        fused_feat = self.fusion(
            point_feat=point_feat,
            image_feat=dino_feat,
            geometric_feat=geo_feat,
        )  # (B, N, fusion_dim)
        
        # 5. Decode to segmentation
        logits = self.decoder(fused_feat)  # (B, N, num_classes)
        
        return logits
    
    def get_predictions(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Get binary predictions for point cloud segmentation.
        
        Args:
            points: (B, N, 3) point coordinates
            colors: (B, N, 3) point colors
            image: (B, 3, H, W) RGB image
            mask: (B, 1, H, W) 2D object mask
            threshold: Classification threshold
        
        Returns:
            predictions: (B, N) binary predictions
        """
        logits = self.forward(points, colors, image, mask)
        probs = torch.sigmoid(logits.squeeze(-1))
        predictions = (probs > threshold).float()
        return predictions
