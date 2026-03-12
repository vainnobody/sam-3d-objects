from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from geoalign.gaussian_features import (
    build_attribute_tensor,
    compute_view_mapping,
    discover_label_file,
    discover_mask_files,
    discover_prior_file,
    extract_gaussian_attributes,
    load_labels,
    load_mask,
    load_scene_and_gaussians,
    render_depth_map,
    _load_optional_array,
)


def main(config) -> None:
    bundle = load_scene_and_gaussians(config)
    cache_cfg = config.geoalign.cache
    cache_root = Path(config.geoalign.cache_dir) / cache_cfg.split / Path(config.scene.scene_path).name
    cache_root.mkdir(parents=True, exist_ok=True)

    mask_root = Path(cache_cfg.mask_root)
    prior_root = Path(cache_cfg.prior_root) if cache_cfg.prior_root else None
    label_root = Path(cache_cfg.label_root) if cache_cfg.label_root else None

    attr = extract_gaussian_attributes(bundle.gaussians)
    scene_gaussian_count = int(attr['xyz'].shape[0])
    index = []

    for view_idx in tqdm(range(len(bundle.train_views)), desc='prepare geoalign cache'):
        view = bundle.train_views[view_idx]
        camera_info = bundle.camera_infos[view_idx]
        image_hw = (int(view.image_height), int(view.image_width))
        depth = render_depth_map(view, bundle.gaussians, config.pipeline, image_hw)
        mapping, _ = compute_view_mapping(
            view,
            camera_info,
            attr['xyz'],
            depth,
            image_hw=image_hw,
            visibility_threshold=cache_cfg.visibility_threshold,
            cut_boundary=cache_cfg.cut_boundary,
        )
        visible = mapping[:, 2].astype(bool)
        visible_indices = torch.from_numpy(np.flatnonzero(visible)).long()
        if visible_indices.numel() == 0:
            continue

        mask_files = discover_mask_files(mask_root, view.image_name)
        for mask_path in mask_files:
            object_name = mask_path.stem
            prior_path = discover_prior_file(prior_root, view.image_name, object_name)
            label_path = discover_label_file(label_root, view.image_name, object_name)

            occ_score = _load_optional_array(prior_path, 'occ_score', scene_gaussian_count)[visible_indices]
            surface_distance = _load_optional_array(prior_path, 'surface_distance', scene_gaussian_count)[visible_indices]
            inside_flag = _load_optional_array(prior_path, 'inside_flag', scene_gaussian_count)[visible_indices]
            depth_residual = _load_optional_array(prior_path, 'depth_residual', scene_gaussian_count)[visible_indices]
            labels = load_labels(label_path, scene_gaussian_count)
            if labels is not None:
                labels = labels[visible_indices]

            sample = {
                'scene_name': Path(config.scene.scene_path).name,
                'image_name': view.image_name,
                'image_path': view.image_path,
                'object_name': object_name,
                'image': view.original_image.cpu(),
                'mask_2d': load_mask(mask_path, image_hw),
                'image_hw': list(image_hw),
                'visible_indices': visible_indices,
                'image_coords': torch.from_numpy(mapping[visible][:, :2]).long(),
                'gaussian_xyz': attr['xyz'][visible_indices].cpu(),
                'gaussian_attr': build_attribute_tensor(attr, visible_indices).cpu(),
                'occ_score': occ_score.cpu(),
                'surface_distance': surface_distance.cpu(),
                'inside_flag': inside_flag.cpu(),
                'depth_residual': depth_residual.cpu(),
                'gaussian_label': labels.cpu() if labels is not None else None,
                'sonata_feat': None,
                'num_scene_gaussians': scene_gaussian_count,
            }
            out_path = cache_root / f'{view.image_name}__{object_name}.pt'
            torch.save(sample, out_path)
            index.append({'image_name': view.image_name, 'object_name': object_name, 'cache_path': str(out_path)})

    with open(cache_root / 'index.json', 'w', encoding='utf-8') as fp:
        json.dump(index, fp, indent=2)


if __name__ == '__main__':
    config = OmegaConf.load('./config/geoalign_base.yaml')
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))
    main(config)
