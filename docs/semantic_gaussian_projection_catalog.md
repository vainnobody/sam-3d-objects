# Semantic Gaussians 几何约束投影条目清单

本清单只保留与 `几何约束投影 / mask 约束 lifting / 3DGS 语义映射` 直接相关的工作。

## A. 基线与高斯特征 lifting

| Title | Year | Type | Core Signal | Code | Relevance | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Semantic Gaussians | 2024 | 3DGS open-vocab scene understanding | 2D semantic feature projection + 3D distillation | <https://github.com/sharinka0715/semantic-gaussians> | Very High | 当前基线；问题就在它的直接投影与平均。 |
| Feature 3DGS | 2024 | 3DGS feature field distillation | Distilled 2D foundation features | <https://github.com/ShijieZhou-UCLA/feature-3dgs> | High | 适合参考 feature-Gaussian 表达与渲染。 |
| FiT3D | 2024 | 3D-aware feature lifting and fine-tuning | 2D features -> 3D Gaussian -> re-rendered 2D features | <https://github.com/ywyue/FiT3D> | High | 强调 3D 几何能反向矫正 2D 特征。 |

## B. 直接处理 2D -> 3DGS segmentation / lifting

| Title | Year | Type | Core Signal | Code | Relevance | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Unified-Lift / Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting | 2025 | Object-aware lifting for 3DGS | Gaussian-level feature + object-level codebook + noisy label filtering | Not found | Very High | 和你的目标最像，但当前未确认公开仓库。 |
| FlashSplat | 2024 | Optimal 2D-to-3DGS segmentation | Global optimal solver for Gaussian labels from 2D masks | <https://github.com/florinshen/flashsplat> | Very High | 适合借鉴 mask-constrained assignment。 |
| Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks | 2024 | Training-free 2D mask backprojection | Voting / masked gradient / backprojection | Project page only: <https://jojijoseph.github.io/3dgs-segmentation/> | High | 适合做 per-view weighting 而不是硬采样。 |
| Gradient-Weighted Feature Back-Projection | 2025 | Training-free feature lifting | Gradient-weighted backprojection | Project page only: <https://jojijoseph.github.io/3dgs-backprojection/> | High | 更贴近 feature projection 而非仅 segmentation。 |

## C. 对象级与多视角一致性

| Title | Year | Type | Core Signal | Code | Relevance | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| OpenMask3D | 2023 | Open-vocab 3D instance segmentation | 3D mask proposal + multi-view CLIP fusion | <https://github.com/OpenMask3D/openmask3d> | Very High | 最值得借鉴的是对象级聚合，而不是点级平均。 |
| Panoptic Lifting | 2023 | Multi-view 2D masks -> 3D consistent panoptic field | Assignment + cross-view consistency | Project page with code links: <https://nihalsid.github.io/panoptic-lifting/> | High | 适合解决 view-wise instance mismatch。 |
| PCF-Lift | 2024 | Probabilistic panoptic lifting | Probabilistic feature fusion under noisy masks | <https://github.com/Runsong123/PCF-Lift> | High | 适合迁移成 uncertainty-aware feature fusion。 |

## D. 与 SAM3D 结合的可迁移点

| Source | What It Provides | How to Use It |
| --- | --- | --- |
| `sam3d_objects` stage-1 inference | pointmap, voxel, shape latent, object geometry proxy | 对投影位置做几何合法性检查，而不是只看深度。 |
| `sam3d_objects` shape latent | coarse volumetric occupancy prior | 为每个高斯提供 object occupancy score。 |
| `sam3d_objects` pointmap / mesh proxy | visible surface support | 用于过滤落在 mask 但不在几何表面的投影。 |

## E. 最值得优先复现的机制

### 1. 对 `fusion.py` 最直接的机制

- FlashSplat 的 mask-to-Gaussian assignment
- Gradient-weighted backprojection
- FiT3D 的 3D-aware feature correction

### 2. 对长期方案最重要的机制

- Unified-Lift 的 object-aware lifting
- OpenMask3D 的 object-centric feature aggregation
- PCF-Lift 的 uncertainty-aware fusion

## F. 来源链接

### Paper links

- Semantic Gaussians: <https://arxiv.org/abs/2403.15624>
- Feature 3DGS: <https://arxiv.org/abs/2312.03203>
- FiT3D: <https://arxiv.org/abs/2407.20229>
- OpenMask3D: <https://arxiv.org/abs/2306.13631>
- Panoptic Lifting: <https://arxiv.org/abs/2212.09802>
- PCF-Lift: <https://arxiv.org/abs/2410.10659>
- Unified-Lift: <https://arxiv.org/abs/2503.14029>
- FlashSplat: <https://arxiv.org/abs/2409.08270>
- Gradient-Driven 3D Segmentation: <https://arxiv.org/abs/2409.11681>

### Code / project links

- Semantic Gaussians: <https://github.com/sharinka0715/semantic-gaussians>
- Feature 3DGS: <https://github.com/ShijieZhou-UCLA/feature-3dgs>
- FiT3D: <https://github.com/ywyue/FiT3D>
- OpenMask3D: <https://github.com/OpenMask3D/openmask3d>
- PCF-Lift: <https://github.com/Runsong123/PCF-Lift>
- FlashSplat: <https://github.com/florinshen/flashsplat>
- Panoptic Lifting project: <https://nihalsid.github.io/panoptic-lifting/>
- Gradient-Driven 3DGS Segmentation project: <https://jojijoseph.github.io/3dgs-segmentation/>
- Gradient-Weighted Feature Back-Projection project: <https://jojijoseph.github.io/3dgs-backprojection/>
