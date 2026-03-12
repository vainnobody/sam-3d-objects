from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from geoalign.dataset import GeoAlignCachedDataset, geoalign_collate
from geoalign.model import GeoAlignMaskModel


if __name__ == '__main__':
    config = OmegaConf.load('./config/geoalign_base.yaml')
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)

    checkpoint = torch.load(config.geoalign.checkpoint, map_location='cpu')
    dataset = GeoAlignCachedDataset(config.geoalign.cache_dir, split=config.geoalign.train_split)
    sample = dataset[config.geoalign.sample_index]
    batch = geoalign_collate([sample])
    batch['image'] = batch['image'].to(config.geoalign.device)
    batch['mask_2d'] = batch['mask_2d'].to(config.geoalign.device)

    model = GeoAlignMaskModel(
        gaussian_attr_dim=checkpoint['gaussian_attr_dim'],
        image_dim=config.geoalign.model.image_dim,
        gaussian_dim=config.geoalign.model.gaussian_dim,
        geometry_dim=config.geoalign.model.geometry_dim,
        sonata_dim=checkpoint.get('sonata_dim', 0),
    ).to(config.geoalign.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with torch.no_grad():
        outputs = model(batch)
    probs = outputs.mask_probs[0].cpu()
    full_mask = torch.zeros(sample['num_scene_gaussians'], dtype=probs.dtype)
    full_mask[sample['visible_indices']] = probs

    out_dir = Path(config.geoalign.output_dir) / 'pred_masks'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample['image_name']}__{sample['object_name']}.pt"
    torch.save(
        {
            'gaussian_mask': (full_mask > config.geoalign.infer.threshold).float(),
            'gaussian_scores': full_mask,
            'visible_indices': sample['visible_indices'],
            'image_name': sample['image_name'],
            'object_name': sample['object_name'],
        },
        out_path,
    )
    print(f'Saved prediction to {out_path}')
