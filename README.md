# SAM3D GeoAlign

Geometry-constrained 3D Gaussian object segmentation built on top of SAM 3D Objects, Semantic Gaussians, and Sonata.

This repository is a research workspace that combines three pieces:
- `sam3d_objects/`: single-image 3D reconstruction utilities from SAM 3D Objects
- `semantic-gaussians/`: a vendored 3D Gaussian Splatting semantic pipeline with camera loading, projection, and rendering
- `sonata/`: optional point-cloud foundation features for stronger 3D conditioning

On top of them, this repo adds a lightweight GeoAlign training path that learns to predict object-level 3D Gaussian masks from:
- an RGB image
- a 2D object mask from SAM or another segmenter
- scene Gaussians from a reconstructed 3DGS scene
- optional SAM3D geometry priors and Sonata features

## What is included

- `scripts/download_objaverse_xl.py`: filtered Objaverse-XL downloader
- `scripts/download_semantic_gaussians_sample.py`: tiny Semantic Gaussians-compatible sample downloader
- `docs/`: survey notes, design notes, and the GeoAlign method spec
- `semantic-gaussians/prepare_geoalign_cache.py`: prepare single-view GeoAlign training samples from 3DGS scenes
- `semantic-gaussians/train_geoalign_mask.py`: train the lightweight Gaussian mask predictor
- `semantic-gaussians/infer_geoalign_mask.py`: infer and export 3D Gaussian masks
- `notebook/`: SAM 3D object demos and alignment notebooks

## Repository layout

```text
sam3d_objects/       Core SAM 3D object reconstruction code
semantic-gaussians/  Vendored Semantic Gaussians code + GeoAlign extensions
sonata/              Optional point-cloud foundation encoder
scripts/             Utility scripts
notebook/            Demo notebooks and helper inference code
docs/                Research survey and implementation notes
```

## Installation

This codebase mixes multiple upstream projects, so installation is split by subsystem.

### 1. Base SAM 3D Objects environment

```bash
python -m pip install -e .
python -m pip install -e '.[dev,inference]'
```

If you prefer the provided Conda environment:

```bash
conda env create -f environments/default.yml
conda activate sam3d-objects
```

### 2. Semantic Gaussians environment

The `semantic-gaussians/` directory keeps its own environment and CUDA-dependent extensions:

```bash
cd semantic-gaussians
conda env create -f environment.yml
conda activate sega
pip install -r requirements.txt
```

You will still need to compile the 3DGS-related submodules required by that project.

### 3. Sonata (optional)

Use this only if you want to add point-cloud foundation features:

```bash
cd sonata
conda env create -f environment.yml
conda activate sonata
# or install as a package
python setup.py install
```

## Quick start

### SAM 3D Objects demo

```bash
python demo.py
```

This runs the reference single-image reconstruction flow and writes `splat.ply`.

### Objaverse-XL download helper

```bash
python scripts/download_objaverse_xl.py --annotations-only
```

### LERF-OVS benchmark downloader

Download one or more LERF-OVS scenes:

```bash
python scripts/download_lerf_ovs.py --scene figurines
python scripts/download_lerf_ovs.py --scene figurines ramen
python scripts/download_lerf_ovs.py --scene all
python scripts/download_lerf_ovs.py --list-scenes
```

By default this extracts scenes into:

```text
data/lerf_ovs/<scene>
```

The upstream release is distributed as a single archive, and the script only
extracts the requested scene directories.

### Tiny Semantic Gaussians test scene

To smoke-test the Semantic Gaussians side of this repo with a small complete scene:

```bash
python scripts/download_semantic_gaussians_sample.py
```

如果你想先只验证下载链路是否工作，而不真的拉完整包：

```bash
python scripts/download_semantic_gaussians_sample.py --test-download-mb 8
```

By default this downloads the `Mip-NeRF 360 / stump` scene into:

```text
data/semantic_gaussians_samples/mipnerf360/stump
```

This scene is small enough for iterative testing and is compatible with the
COLMAP-style loaders used by `semantic-gaussians/`.

### GeoAlign workflow

1. Train or prepare a 3DGS scene under `semantic-gaussians/`.
2. Prepare per-view 2D masks.
3. Optionally export SAM3D geometry priors per object/view.
4. Build the GeoAlign cache.
5. Train the Gaussian mask model.
6. Run inference to export 3D Gaussian masks.

Example commands:

```bash
cd semantic-gaussians
python prepare_geoalign_cache.py
python train_geoalign_mask.py
python infer_geoalign_mask.py
```

Before running them, update `semantic-gaussians/config/geoalign_base.yaml` to point at your scene, 3DGS model, masks, and optional priors.

## GeoAlign method

The current GeoAlign path targets lightweight single-view conditional 3D Gaussian segmentation.

Model input:
- image
- binary 2D object mask
- visible scene Gaussians and their attributes
- optional SAM3D occupancy / surface priors
- optional Sonata features

Model output:
- per-Gaussian object logits
- per-Gaussian object probabilities
- exported 3D Gaussian mask for downstream rendering or editing

Additional design notes live in:
- `docs/geoalign_lift_idea.md`
- `docs/geoalign_lift_lite_spec.md`
- `docs/semantic_gaussian_projection_design_notes.md`

## Data and weights

This repository does not include:
- model checkpoints
- datasets
- local experiment outputs
- cached downloads

You must prepare them locally and point configs to the correct paths.

## Upstream projects and attribution

This repository contains or builds on code and ideas from:
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [Semantic Gaussians](https://github.com/sharinka0715/semantic-gaussians)
- [Sonata](https://github.com/Pointcept/Sonata)

Please consult the corresponding subdirectories and upstream repositories for licensing details and original setup instructions.

## License

The root project keeps the original `LICENSE` file from SAM 3D Objects. Vendored subprojects may have additional license terms in their own directories. Make sure your usage complies with all applicable upstream licenses.
