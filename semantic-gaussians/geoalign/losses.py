from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import render_chn


@dataclass
class LossBreakdown:
    total: torch.Tensor
    mask: torch.Tensor
    geometry: torch.Tensor
    reprojection: torch.Tensor
    sparse: torch.Tensor


def dice_loss(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    prob = prob.reshape(-1)
    target = target.reshape(-1)
    intersection = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


class GeoAlignLoss(nn.Module):
    def __init__(self, lambda_mask: float = 1.0, lambda_geo: float = 0.2, lambda_reproj: float = 0.5, lambda_sparse: float = 0.05):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_geo = lambda_geo
        self.lambda_reproj = lambda_reproj
        self.lambda_sparse = lambda_sparse

    def _render_soft_mask(self, batch_item_idx: int, probs: torch.Tensor, batch: dict[str, Any], render_ctx: dict[str, Any]) -> torch.Tensor:
        scene_gaussians = render_ctx['gaussians']
        camera = render_ctx['camera_map'][batch['image_name'][batch_item_idx]]
        camera.cuda()
        full_scores = torch.zeros(
            (scene_gaussians.get_xyz.shape[0], 1),
            dtype=probs.dtype,
            device=probs.device,
        )
        visible_indices = batch['visible_indices'][batch_item_idx].to(probs.device)
        full_scores[visible_indices] = probs.unsqueeze(-1)
        background = torch.zeros(1, dtype=probs.dtype, device=probs.device)
        render_pkg = render_chn(
            camera,
            scene_gaussians,
            render_ctx['pipeline'],
            background,
            num_channels=1,
            override_color=full_scores,
            override_shape=(int(batch['image_hw'][batch_item_idx][1]), int(batch['image_hw'][batch_item_idx][0])),
        )
        return render_pkg['render'].clamp(0.0, 1.0)

    def _fallback_project(self, batch_item_idx: int, probs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        coords = batch['image_coords'][batch_item_idx].to(probs.device).long()
        h, w = map(int, batch['image_hw'][batch_item_idx])
        heatmap = torch.zeros((1, h, w), dtype=probs.dtype, device=probs.device)
        if coords.numel() == 0:
            return heatmap
        y = coords[:, 0].clamp(0, h - 1)
        x = coords[:, 1].clamp(0, w - 1)
        flat = y * w + x
        heatmap.view(-1).index_add_(0, flat, probs)
        return heatmap / heatmap.amax().clamp_min(1.0)

    def forward(self, outputs: Any, batch: dict[str, Any], render_ctx: dict[str, Any] | None = None) -> LossBreakdown:
        mask_losses = []
        geometry_losses = []
        reproj_losses = []
        sparse_losses = []
        labels = batch.get('gaussian_label')

        for i, logits in enumerate(outputs.mask_logits):
            probs = outputs.mask_probs[i]
            if labels is not None and labels[i] is not None:
                target = labels[i].to(logits.device)
                mask_losses.append(F.binary_cross_entropy_with_logits(logits, target) + dice_loss(probs, target))
            else:
                mask_losses.append(logits.new_tensor(0.0))

            occ_score = batch['occ_score'][i].to(logits.device).squeeze(-1)
            surface_distance = batch['surface_distance'][i].to(logits.device).squeeze(-1)
            inside_flag = batch['inside_flag'][i].to(logits.device).squeeze(-1)
            depth_residual = batch['depth_residual'][i].to(logits.device).squeeze(-1)
            geometry = (
                probs * (1.0 - occ_score)
                + probs * surface_distance.clamp_min(0.0)
                + probs * (1.0 - inside_flag)
                + probs * depth_residual.abs()
            ).mean()
            geometry_losses.append(geometry)

            if render_ctx is not None:
                reproj_mask = self._render_soft_mask(i, probs, batch, render_ctx)
            else:
                reproj_mask = self._fallback_project(i, probs, batch)
            target_2d = batch['mask_2d'][i].to(logits.device)
            reproj_losses.append(F.binary_cross_entropy(reproj_mask, target_2d) + dice_loss(reproj_mask, target_2d))
            sparse_losses.append(probs.mean())

        mask_loss = torch.stack(mask_losses).mean()
        geometry_loss = torch.stack(geometry_losses).mean()
        reprojection_loss = torch.stack(reproj_losses).mean()
        sparse_loss = torch.stack(sparse_losses).mean()
        total = (
            self.lambda_mask * mask_loss
            + self.lambda_geo * geometry_loss
            + self.lambda_reproj * reprojection_loss
            + self.lambda_sparse * sparse_loss
        )
        return LossBreakdown(total, mask_loss, geometry_loss, reprojection_loss, sparse_loss)
