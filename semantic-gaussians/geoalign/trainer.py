from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import GeoAlignCachedDataset, geoalign_collate
from .gaussian_features import load_scene_and_gaussians
from .losses import GeoAlignLoss
from .model import GeoAlignMaskModel


def _build_camera_map(train_views) -> dict[str, object]:
    camera_map = {}
    for idx in range(len(train_views)):
        camera = train_views[idx]
        camera_map[camera.image_name] = camera
    return camera_map


def _prepare_output_dir(config) -> Path:
    output_dir = Path(config.geoalign.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    with open(output_dir / 'geoalign_config.yaml', 'w', encoding='utf-8') as fp:
        OmegaConf.save(config, fp)
    return output_dir


def _infer_model_dims(dataset: GeoAlignCachedDataset, config) -> tuple[int, int]:
    sample = dataset[0]
    gaussian_attr_dim = int(sample['gaussian_attr'].shape[-1])
    sonata_dim = 0
    if sample.get('sonata_feat') is not None:
        sonata_dim = int(sample['sonata_feat'].shape[-1])
    if config.geoalign.model.sonata_dim > 0:
        sonata_dim = int(config.geoalign.model.sonata_dim)
    return gaussian_attr_dim, sonata_dim


def train_geoalign(config) -> None:
    bundle = load_scene_and_gaussians(config)
    train_dataset = GeoAlignCachedDataset(config.geoalign.cache_dir, split=config.geoalign.train_split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.geoalign.train.batch_size,
        shuffle=True,
        num_workers=config.geoalign.train.num_workers,
        collate_fn=geoalign_collate,
    )

    gaussian_attr_dim, sonata_dim = _infer_model_dims(train_dataset, config)
    model = GeoAlignMaskModel(
        gaussian_attr_dim=gaussian_attr_dim,
        image_dim=config.geoalign.model.image_dim,
        gaussian_dim=config.geoalign.model.gaussian_dim,
        geometry_dim=config.geoalign.model.geometry_dim,
        sonata_dim=sonata_dim,
    ).to(config.geoalign.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.geoalign.train.lr,
        weight_decay=config.geoalign.train.weight_decay,
    )
    criterion = GeoAlignLoss(
        lambda_mask=config.geoalign.loss.mask_w,
        lambda_geo=config.geoalign.loss.geo_w,
        lambda_reproj=config.geoalign.loss.reproj_w,
        lambda_sparse=config.geoalign.loss.sparse_w,
    )

    render_ctx = {
        'gaussians': bundle.gaussians,
        'pipeline': config.pipeline,
        'camera_map': _build_camera_map(bundle.train_views),
    }
    output_dir = _prepare_output_dir(config)
    global_step = 0

    for epoch in range(config.geoalign.train.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f'geoalign epoch {epoch + 1}/{config.geoalign.train.epochs}')
        for batch in progress:
            batch['image'] = batch['image'].to(config.geoalign.device)
            batch['mask_2d'] = batch['mask_2d'].to(config.geoalign.device)
            outputs = model(batch)
            loss_dict = criterion(outputs, batch, render_ctx=render_ctx)
            optimizer.zero_grad(set_to_none=True)
            loss_dict.total.backward()
            optimizer.step()

            global_step += 1
            progress.set_postfix(
                loss=f'{loss_dict.total.item():.4f}',
                reproj=f'{loss_dict.reprojection.item():.4f}',
                geo=f'{loss_dict.geometry.item():.4f}',
            )

            if global_step % config.geoalign.train.save_every == 0:
                save_path = output_dir / 'checkpoints' / f'step_{global_step:06d}.pt'
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'global_step': global_step,
                        'config': OmegaConf.to_container(config, resolve=True),
                        'gaussian_attr_dim': gaussian_attr_dim,
                        'sonata_dim': sonata_dim,
                    },
                    save_path,
                )

    final_path = output_dir / 'checkpoints' / 'last.pt'
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step,
            'config': OmegaConf.to_container(config, resolve=True),
            'gaussian_attr_dim': gaussian_attr_dim,
            'sonata_dim': sonata_dim,
        },
        final_path,
    )
