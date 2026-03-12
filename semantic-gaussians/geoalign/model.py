from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feat = self.net(image)
        pooled_mask = F.interpolate(mask, size=feat.shape[-2:], mode='nearest')
        denom = pooled_mask.sum(dim=(2, 3), keepdim=False).clamp_min(1.0)
        return (feat * pooled_mask).sum(dim=(2, 3)) / denom


class GaussianFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, xyz: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([xyz, attr], dim=-1))


class GeometryEncoder(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.net(geometry)


class SonataAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FusionHead(nn.Module):
    def __init__(self, token_dim: int, gaussian_dim: int, geometry_dim: int, sonata_dim: int = 0, hidden_dim: int = 160, num_heads: int = 4):
        super().__init__()
        fused_in_dim = gaussian_dim + geometry_dim + sonata_dim
        self.token_to_gate = nn.Linear(token_dim, hidden_dim)
        self.pre = nn.Linear(fused_in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, object_token: torch.Tensor, gaussian_feat: torch.Tensor, geometry_feat: torch.Tensor, sonata_feat: torch.Tensor | None = None) -> torch.Tensor:
        parts = [gaussian_feat, geometry_feat]
        if sonata_feat is not None:
            parts.append(sonata_feat)
        fused = self.pre(torch.cat(parts, dim=-1))
        gate = torch.sigmoid(self.token_to_gate(object_token)).unsqueeze(1)
        fused = fused * gate
        query = object_token.unsqueeze(1).expand(-1, fused.shape[1], -1)
        if query.shape[-1] != fused.shape[-1]:
            query = F.pad(query, (0, fused.shape[-1] - query.shape[-1]))
        attn_out, _ = self.attn(query, fused, fused)
        fused = self.norm(fused + attn_out)
        return self.decoder(fused).squeeze(-1)


@dataclass
class GeoAlignOutput:
    mask_logits: list[torch.Tensor]
    mask_probs: list[torch.Tensor]


class GeoAlignMaskModel(nn.Module):
    def __init__(self, gaussian_attr_dim: int, image_dim: int = 128, gaussian_dim: int = 128, geometry_dim: int = 32, sonata_dim: int = 0) -> None:
        super().__init__()
        self.image_encoder = ConvStem(out_dim=image_dim)
        self.gaussian_encoder = GaussianFeatureEncoder(gaussian_attr_dim, hidden_dim=gaussian_dim)
        self.geometry_encoder = GeometryEncoder(hidden_dim=geometry_dim)
        self.sonata_adapter = SonataAdapter(sonata_dim, output_dim=32) if sonata_dim > 0 else None
        self.fusion_head = FusionHead(image_dim, gaussian_dim, geometry_dim, sonata_dim=32 if sonata_dim > 0 else 0)

    def forward(self, batch: dict[str, object]) -> GeoAlignOutput:
        image = batch['image']
        mask_2d = batch['mask_2d']
        object_tokens = self.image_encoder(image, mask_2d)
        logits = []
        probs = []
        sonata_values = batch.get('sonata_feat')

        for idx in range(image.shape[0]):
            xyz = batch['gaussian_xyz'][idx].to(image.device)
            attr = batch['gaussian_attr'][idx].to(image.device)
            geometry = torch.cat(
                [
                    batch['occ_score'][idx].to(image.device),
                    batch['surface_distance'][idx].to(image.device),
                    batch['inside_flag'][idx].to(image.device),
                    batch['depth_residual'][idx].to(image.device),
                ],
                dim=-1,
            )
            gaussian_feat = self.gaussian_encoder(xyz, attr).unsqueeze(0)
            geometry_feat = self.geometry_encoder(geometry).unsqueeze(0)
            sonata_feat = None
            if self.sonata_adapter is not None and sonata_values[idx] is not None:
                sonata_feat = self.sonata_adapter(sonata_values[idx].to(image.device)).unsqueeze(0)
            logit = self.fusion_head(object_tokens[idx : idx + 1], gaussian_feat, geometry_feat, sonata_feat).squeeze(0)
            logits.append(logit)
            probs.append(torch.sigmoid(logit))
        return GeoAlignOutput(mask_logits=logits, mask_probs=probs)
