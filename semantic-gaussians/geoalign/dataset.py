from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class GeoAlignCachedDataset(Dataset):
    def __init__(self, cache_root: str, split: str = 'train', scene_name: str | None = None) -> None:
        self.cache_root = Path(cache_root)
        split_root = self.cache_root / split
        if split_root.is_dir():
            sample_files = sorted(split_root.rglob('*.pt'))
        else:
            sample_files = sorted(self.cache_root.rglob('*.pt'))
        if scene_name is not None:
            sample_files = [path for path in sample_files if path.parent.name == scene_name or scene_name in str(path)]
        if not sample_files:
            raise RuntimeError(f'No cached GeoAlign samples found in {split_root}')
        self.sample_files = sample_files

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = torch.load(self.sample_files[index], map_location='cpu')
        sample['cache_path'] = str(self.sample_files[index])
        return sample


def geoalign_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    tensor_keys = {'image', 'mask_2d'}
    list_keys = {
        'visible_indices',
        'gaussian_xyz',
        'gaussian_attr',
        'occ_score',
        'surface_distance',
        'inside_flag',
        'depth_residual',
        'gaussian_label',
        'sonata_feat',
        'image_coords',
    }
    for key in batch[0].keys():
        values = [item.get(key) for item in batch]
        if key in tensor_keys:
            collated[key] = torch.stack(values, dim=0)
        elif key in list_keys:
            collated[key] = values
        else:
            collated[key] = values
    return collated
