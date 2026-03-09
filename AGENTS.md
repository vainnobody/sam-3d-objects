# SAM 3D Objects - AI Agent 上下文指南

> 本文件为 AI Agent 提供项目的全面上下文信息，用于辅助代码理解、修改和开发。

---

## 目录

1. [项目概述](#项目概述)
2. [架构详解](#架构详解)
3. [构建与运行](#构建与运行)
4. [开发约定](#开发约定)
5. [GPU 要求](#gpu-要求)
6. [故障排除](#故障排除)
7. [贡献指南](#贡献指南)
8. [关键文件索引](#关键文件索引)

---

## 项目概述

### 目的与功能

**SAM 3D Objects** 是 Meta AI 开发的基础模型，用于从单张图像重建完整的 3D 形状几何、纹理和布局。

**核心能力**:
- 单图像 3D 重建：从单张图片重建物体的 3D 模型（姿态、形状、纹理、布局）
- 多物体支持：可同时处理图像中的多个物体
- 遮挡处理：能够处理现实世界中具有遮挡和杂乱场景的情况
- 渐进式训练：使用带人类反馈的数据引擎进行训练
- 输出格式：导出为 Gaussian Splat (`.ply` 文件) 或网格格式

### 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.5.1, Lightning 2.3.3 |
| 3D 处理 | PyTorch3D, Kaolin 0.17.0, Open3D 0.18.0 |
| CUDA 版本 | CUDA 12.1 |
| Python 版本 | Python 3.11.0 |
| 配置管理 | Hydra Core 1.3.2, OmegaConf |
| 深度估计 | MoGe (Microsoft) |
| 高斯渲染 | gsplat, NVDiffRast |
| 可视化 | Polyscope, Plotly, Gradio |

### 硬件要求

| 要求 | 规格 |
|------|------|
| 操作系统 | Linux 64 位架构 |
| GPU | NVIDIA GPU，至少 32GB 显存 |
| 推荐显卡 | A100 40GB/80GB, H100, RTX 6000 Ada |

---

## 架构详解

### 目录结构

```
sam-3d-objects/
├── sam3d_objects/                 # 核心源代码
│   ├── model/                     # 模型架构
│   │   ├── io.py                  # 模型加载/保存
│   │   ├── backbone/              # 主干网络
│   │   │   ├── dit/               # Diffusion Transformer
│   │   │   │   └── embedder/      # 条件嵌入器 (DINO, PointMap)
│   │   │   ├── generator/         # 生成器 (CFG, Flow Matching)
│   │   │   └── tdfy_dit/          # 3D Diffusion Transformer
│   │   │       ├── models/        # 稀疏结构流/VAE
│   │   │       ├── modules/       # 注意力/稀疏张量
│   │   │       └── renderers/     # 高斯渲染器
│   │   └── layers/                # 神经网络层
│   │       └── llama3/            # LLaMA 前馈层
│   ├── pipeline/                  # 推理管道
│   │   ├── inference_pipeline.py  # 基础推理管道
│   │   ├── inference_pipeline_pointmap.py  # 点图推理管道
│   │   ├── inference_utils.py     # 推理工具函数
│   │   ├── preprocess_utils.py    # 预处理工具
│   │   ├── layout_post_optimization_utils.py  # 布局后优化
│   │   ├── depth_models/          # 深度估计模型
│   │   └── utils/                 # 管道工具
│   ├── data/                      # 数据处理
│   │   ├── utils.py               # 数据工具函数
│   │   └── dataset/tdfy/          # TDFY 数据集处理
│   │       ├── preprocessor.py    # 预处理器
│   │       ├── transforms_3d.py   # 3D 变换
│   │       └── pose_target.py     # 姿态目标转换
│   ├── config/                    # 配置管理
│   │   └── utils.py               # 配置工具
│   └── utils/                     # 工具函数
│       └── visualization/         # 可视化工具
├── notebook/                      # Jupyter Notebooks 和示例
│   ├── inference.py               # 公开推理 API
│   ├── demo_single_object.ipynb   # 单物体重建示例
│   ├── demo_multi_object.ipynb    # 多物体重建示例
│   ├── demo_3db_mesh_alignment.ipynb  # SAM 3D Body 对齐
│   ├── mesh_alignment.py          # 网格对齐工具
│   ├── images/                    # 示例图像数据集
│   ├── gaussians/                 # 输出 Gaussian Splat
│   └── meshes/                    # 输出网格文件
├── checkpoints/                   # 模型权重目录
├── environments/                  # Conda 环境配置
│   └── default.yml                # 默认环境配置
├── patching/                      # 补丁脚本
│   └── hydra                      # Hydra 配置补丁
├── doc/                           # 文档和图片
├── demo.py                        # 快速演示脚本
└── pyproject.toml                 # 项目配置
```

### 核心模块职责

#### 1. 推理管道 (`sam3d_objects/pipeline/`)

**继承关系**:
```
InferencePipeline (基类)
    └── InferencePipelinePointMap (支持深度图/点图输入)
```

**两阶段生成流程**:

```
输入图像 + 掩码
       │
       ▼
┌──────────────────────────────────────┐
│  阶段 1: 稀疏结构生成 (Sparse Structure) │
│  - DINO 特征提取                       │
│  - Diffusion Transformer              │
│  - 输出: 稀疏坐标 + 特征               │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  阶段 2: 结构化潜在生成 (Structured Latent) │
│  - 点图嵌入 (可选)                    │
│  - 稀疏到密集解码                     │
│  - 输出: Gaussian Splat 参数          │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  后处理                               │
│  - 布局优化 (可选)                    │
│  - 网格生成 (可选)                    │
│  - 纹理烘焙 (可选)                    │
└──────────────────────────────────────┘
       │
       ▼
输出: Gaussian Splat (.ply) 或 Mesh
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ss_inference_steps` | 25 | 稀疏结构采样步数 |
| `slat_inference_steps` | 25 | 结构化潜在采样步数 |
| `ss_cfg_strength` | 7 | 阶段 1 CFG 强度 |
| `slat_cfg_strength` | 5 | 阶段 2 CFG 强度 |

#### 2. 条件嵌入器 (`sam3d_objects/model/backbone/dit/embedder/`)

**DINO 嵌入器** (`dino.py`):
- 使用 DINOv2 提取图像特征
- 支持 `dinov2_vitb14`, `dinov2_vitl14` 等变体

**点图嵌入器** (`pointmap.py`):
- 将 (x, y, z) 坐标投影到嵌入空间
- 分割为 patches，每个 patch 内运行自注意力
- 用于场景布局估计

#### 3. Classifier-Free Guidance (`sam3d_objects/model/backbone/generator/classifier_free_guidance.py`)

**无条件处理类型**:
- `zeros`: 将条件张量置零
- `discard`: 丢弃条件参数
- `drop_tensors`: 丢弃张量但保留非张量
- `add_flag`: 添加 CFG 标志

**点图 CFG 变体** (`PointmapCFG`):
```python
# 基于论文 https://arxiv.org/abs/2411.18613
output = y_cond + strength_pm * (y_cond - y_unpm) + strength * (y_unpm - y_uncond)
```

#### 4. 布局后优化 (`sam3d_objects/pipeline/layout_post_optimization_utils.py`)

**三阶段优化流程**:
1. **手动对齐**: 基于点云的高度和中心对齐
2. **ICP 对齐**: 使用 Open3D 进行点云配准
3. **渲染比较**: 基于渲染结果的优化

### 算法实现位置索引

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| 稀疏结构采样 | `sam3d_objects/pipeline/inference_pipeline.py` | 519-565 |
| 结构化潜在采样 | `sam3d_objects/pipeline/inference_pipeline.py` | 567-604 |
| 姿态解码 | `sam3d_objects/pipeline/inference_utils.py` | 435-500 |
| ICP 对齐 | `sam3d_objects/pipeline/layout_post_optimization_utils.py` | 314-337 |
| 点图归一化 | `sam3d_objects/pipeline/utils/pointmap.py` | 17-82 |
| 相机内参推断 | `sam3d_objects/pipeline/utils/pointmap.py` | 17-82 |
| Gaussian 渲染 | `sam3d_objects/model/backbone/tdfy_dit/renderers/` | - |
| CFG 计算 | `sam3d_objects/model/backbone/generator/classifier_free_guidance.py` | 26-59 |

---

## 构建与运行

### 环境设置

#### 1. 创建 Conda 环境

```bash
mamba env create -f environments/default.yml
mamba activate sam3d-objects
```

**环境配置详情** (`environments/default.yml`):
- 环境名称: `sam3d-objects`
- Python 版本: `3.11.0`
- CUDA 版本: `12.1`
- 包含完整的 CUDA Toolkit 和编译工具链

#### 2. 设置 PyTorch/CUDA 依赖源

```bash
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
```

#### 3. 安装核心依赖

```bash
# 安装开发依赖
pip install -e '.[dev]'

# 安装 PyTorch3D 依赖
pip install -e '.[p3d]'
```

#### 4. 安装推理依赖

```bash
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
```

#### 5. 应用补丁

```bash
./patching/hydra
```

#### 6. 下载模型权重

**需要 HuggingFace 认证**:

```bash
# 登录 HuggingFace
huggingface-cli login

# 下载模型权重
hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
```

**手动下载**: 从 https://huggingface.co/facebook/sam-3d-objects 下载并放置到 `checkpoints/` 目录。

### 使用示例

#### 快速开始 (单物体)

```python
import sys
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# 加载模型
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# 加载图像和掩码
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# 运行推理
output = inference(image, mask, seed=42)

# 导出 Gaussian Splat
output["gs"].save_ply("output/splat.ply")
```

#### 多物体重建

```python
from inference import Inference, load_image, load_masks, make_scene

# 加载所有掩码
masks = load_masks("path/to/images/folder")

# 分别推理每个物体
outputs = []
for mask in masks:
    output = inference(image, mask, seed=42)
    outputs.append(output)

# 合并为场景
scene_gs = make_scene(*outputs)
scene_gs.save_ply("output/scene.ply")
```

#### 视频渲染

```python
from inference import render_video, ready_gaussian_for_video_rendering

# 准备 Gaussian 用于渲染
scene_gs = ready_gaussian_for_video_rendering(output["gs"])

# 渲染视频
frames = render_video(
    scene_gs,
    resolution=512,
    num_frames=300,
    r=2.0,
    fov=40,
)

# 保存为视频
import imageio
imageio.mimsave("output/rotate.mp4", frames, fps=30)
```

#### 运行 Jupyter Notebook

```bash
jupyter notebook
# 打开 notebook/demo_single_object.ipynb 或 demo_multi_object.ipynb
```

---

## 开发约定

### 代码风格

#### 1. 版权声明

所有源文件以 Meta Platforms 版权声明开头:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
```

#### 2. 类型注解

广泛使用 Python 类型注解:

```python
def get_child(obj: Any, *keys: Iterable[Any]) -> Any:
    ...

def load_image(path: str) -> np.ndarray:
    ...
```

#### 3. 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 函数/方法 | snake_case | `load_image`, `run_inference` |
| 类 | PascalCase | `InferencePipeline`, `DinoEmbedder` |
| 常量 | UPPER_SNAKE_CASE | `WHITELIST_FILTERS` |
| 私有方法 | _前缀 | `_cfg_step_tensor` |
| 模块变量 | 单下划线前缀表示内部使用 | `_pipeline` |

#### 4. 文档字符串

使用简洁的文档字符串:

```python
class PointPatchEmbed(nn.Module):
    """
    将(x,y,z)坐标投影到嵌入空间
    分割为patches，每个patch内运行自注意力
    返回每个窗口一个token
    """
```

### 配置管理

#### Hydra + OmegaConf

项目使用 Hydra 进行配置管理:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

# 加载配置
config = OmegaConf.load(config_path)

# 实例化对象
pipeline = instantiate(config)
```

#### 安全机制

Hydra 实例化有白名单/黑名单过滤:

```python
WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target) in {os.system, os.remove, ...},
]
```

**警告**: 修改配置时需确保目标在白名单中。

### 日志规范

使用 `loguru` 进行日志记录:

```python
from loguru import logger

logger.info(f"Loading model weights...")
logger.warning(f"Low memory detected")
logger.error(f"Failed to load checkpoint")
```

### 导入规范

```python
# 标准库
import os
import sys
from typing import Optional, List

# 第三方库
import torch
import numpy as np
from PIL import Image

# 本地模块
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
```

---

## GPU 要求

### 最小硬件配置

| 组件 | 要求 | 推荐 |
|------|------|------|
| GPU | NVIDIA 32GB 显存 | A100 40GB+ |
| CPU | 8 核 | 16 核+ |
| 内存 | 32GB | 64GB+ |
| 存储 | 50GB SSD | 100GB NVMe |

### CUDA 版本兼容性

| CUDA 版本 | PyTorch 版本 | 状态 |
|-----------|--------------|------|
| 12.1 | 2.5.1 | ✅ 官方支持 |
| 12.0 | 2.5.x | ⚠️ 可能兼容 |
| 11.8 | 2.4.x | ❌ 不推荐 |

### 性能优化建议

#### 1. 启用 torch.compile

```python
inference = Inference(config_path, compile=True)
```

#### 2. 减少推理步数

```python
# 快速模式 (质量降低)
output = inference(
    image, mask,
    stage1_inference_steps=10,  # 默认 25
)
```

#### 3. 混合精度推理

```python
# 在 pipeline.yaml 中配置
model:
  dtype: float16
```

#### 4. 内存优化

```python
# 清理缓存
torch.cuda.empty_cache()

# 分批处理多物体
for mask in masks:
    output = inference(image, mask)
    output["gs"].save_ply(f"output/{i}.ply")
    del output
    torch.cuda.empty_cache()
```

---

## 故障排除

### 常见安装问题

#### 问题 1: Kaolin 安装失败

**错误信息**:
```
ERROR: Could not find a version that satisfies the requirement kaolin
```

**解决方案**:
```bash
# 确保 FIND_LINKS 环境变量已设置
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install kaolin==0.17.0
```

#### 问题 2: PyTorch3D 编译错误

**错误信息**:
```
RuntimeError: CUDA_HOME not set
```

**解决方案**:
```bash
# 设置 CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX
# 或
export CUDA_HOME=/usr/local/cuda-12.1
```

#### 问题 3: Flash Attention 安装失败

**解决方案**:
```bash
# 确保使用正确的 CUDA 版本
pip install flash-attn==2.8.3 --no-build-isolation
```

### 运行时错误

#### 错误 1: CUDA 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 减小批处理大小
2. 使用更低的推理步数
3. 启用 CPU 卸载（如果支持）

```python
# 减少内存使用
torch.cuda.set_per_process_memory_fraction(0.8, 0)
```

#### 错误 2: Hydra 目标不允许

**错误信息**:
```
RuntimeError: target 'xxx' is not allowed to be hydra instantiated
```

**解决方案**:
将目标模块添加到白名单 (`notebook/inference.py`):
```python
WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "your_module"},
]
```

#### 错误 3: 模型权重未找到

**错误信息**:
```
FileNotFoundError: checkpoints/hf/pipeline.yaml
```

**解决方案**:
```bash
# 检查权重是否下载
ls checkpoints/

# 重新下载
hf download --repo-type model --local-dir checkpoints/hf --max-workers 1 facebook/sam-3d-objects
```

#### 错误 4: DINO 模型加载失败

**错误信息**:
```
RuntimeError: Failed to load DINO model
```

**解决方案**:
```python
# 手动下载 DINO 权重
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 检查 GPU 状态

```python
import torch
print(f"GPU 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"当前 GPU: {torch.cuda.current_device()}")
print(f"显存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

#### 3. 可视化中间结果

```python
# 可视化掩码
from inference import display_image
display_image(image, masks=[mask])

# 可视化点云
from sam3d_objects.utils.visualization import SceneVisualizer
visualizer = SceneVisualizer()
visualizer.show_pointcloud(output["gaussian"][0].get_xyz)
```

---

## 贡献指南

### 开发环境设置

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/YOUR_USERNAME/sam-3d-objects.git
cd sam-3d-objects

# 2. 创建开发环境
mamba env create -f environments/default.yml
mamba activate sam3d-objects

# 3. 安装开发依赖
pip install -e '.[dev]'

# 4. 安装 pre-commit 钩子 (如果有)
pre-commit install
```

### 代码规范

#### 1. 代码风格检查

```bash
# 使用 black 格式化
black sam3d_objects/

# 使用 isort 排序导入
isort sam3d_objects/

# 使用 flake8 检查
flake8 sam3d_objects/
```

#### 2. 类型检查

```bash
# 使用 mypy 进行类型检查
mypy sam3d_objects/
```

#### 3. 测试

```bash
# 运行测试
pytest tests/

# 运行特定测试
pytest tests/test_pipeline.py -v
```

### Pull Request 流程

1. **创建功能分支**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **提交更改**:
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   ```

   **提交信息规范**:
   - `feat:` 新功能
   - `fix:` 错误修复
   - `docs:` 文档更新
   - `style:` 代码格式调整
   - `refactor:` 代码重构
   - `test:` 测试相关
   - `chore:` 构建/工具相关

3. **推送并创建 PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **PR 检查清单**:
   - [ ] 代码通过所有测试
   - [ ] 代码风格符合规范
   - [ ] 添加了必要的文档
   - [ ] 更新了相关 CHANGELOG
   - [ ] PR 描述清晰说明了更改内容

### 报告问题

在 GitHub Issues 中报告问题时，请包含：

1. **环境信息**:
   - Python 版本
   - PyTorch 版本
   - CUDA 版本
   - GPU 型号

2. **复现步骤**:
   ```python
   # 最小复现代码
   ```

3. **预期行为 vs 实际行为**

4. **错误日志**:
   ```
   完整的错误堆栈
   ```

---

## 关键文件索引

### 核心入口文件

| 文件 | 用途 |
|------|------|
| `demo.py` | 快速演示脚本 |
| `notebook/inference.py` | 公开推理 API |
| `sam3d_objects/__init__.py` | 包初始化 |

### 配置文件

| 文件 | 用途 |
|------|------|
| `pyproject.toml` | 项目依赖和构建配置 |
| `environments/default.yml` | Conda 环境配置 |
| `requirements.txt` | 核心依赖 |
| `requirements.inference.txt` | 推理依赖 |
| `requirements.p3d.txt` | PyTorch3D 依赖 |
| `requirements.dev.txt` | 开发依赖 |
| `checkpoints/hf/pipeline.yaml` | 模型配置 |

### 示例文件

| 文件 | 用途 |
|------|------|
| `notebook/demo_single_object.ipynb` | 单物体重建示例 |
| `notebook/demo_multi_object.ipynb` | 多物体重建示例 |
| `notebook/demo_3db_mesh_alignment.ipynb` | SAM 3D Body 对齐示例 |

### 核心实现文件

| 文件 | 功能 |
|------|------|
| `sam3d_objects/pipeline/inference_pipeline.py` | 基础推理管道 |
| `sam3d_objects/pipeline/inference_pipeline_pointmap.py` | 点图推理管道 |
| `sam3d_objects/model/backbone/tdfy_dit/` | 3D Diffusion Transformer |
| `sam3d_objects/model/backbone/generator/classifier_free_guidance.py` | CFG 实现 |
| `sam3d_objects/pipeline/layout_post_optimization_utils.py` | 布局优化 |

---

## 快速参考

### 常用命令

```bash
# 环境设置
mamba env create -f environments/default.yml
mamba activate sam3d-objects

# 安装
pip install -e '.[dev]' && pip install -e '.[p3d]' && pip install -e '.[inference]'

# 应用补丁
./patching/hydra

# 下载模型
hf download --repo-type model --local-dir checkpoints/hf facebook/sam-3d-objects

# 运行演示
python demo.py

# 运行测试
pytest tests/
```

### 常用 API

```python
from inference import Inference, load_image, load_single_mask, load_masks

# 单物体推理
inference = Inference(config_path)
output = inference(image, mask, seed=42)
output["gs"].save_ply("output.ply")

# 多物体推理
masks = load_masks("path/to/folder")
outputs = [inference(image, mask) for mask in masks]
```

---

*本文件由 AI Agent 自动生成，基于项目代码分析。如有疑问或需要更新，请参考项目文档或联系维护者。*
