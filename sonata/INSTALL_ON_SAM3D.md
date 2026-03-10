# Sonata 安装指南 - 基于 SAM-3D-Objects 环境

> 本文档介绍如何在已有的 `sam3d-objects` conda 环境基础上安装 Sonata 依赖。

---

## 目录

1. [环境差异对比](#1-环境差异对比)
2. [安装方案](#2-安装方案)
3. [方案A: 在现有环境中安装](#3-方案a-在现有环境中安装)
4. [方案B: 创建独立环境 (推荐)](#4-方案b-创建独立环境-推荐)
5. [验证安装](#5-验证安装)
6. [常见问题](#6-常见问题)

---

## 1. 环境差异对比

| 组件 | SAM-3D-Objects | Sonata | 兼容性 |
|------|----------------|--------|--------|
| Python | 3.11 | 3.10 | ⚠️ 小版本差异 |
| CUDA | 12.1 | 12.4 | ⚠️ 需要调整 |
| PyTorch | 2.5.1+cu121 | 2.5.0+cu124 | ✅ 兼容 |
| GCC | 12.4 | 13.2 | ✅ 兼容 |
| CUDA Toolkit | 12.1 | 12.4 | ⚠️ 需要调整 |

**关键依赖差异**:

| 依赖 | SAM-3D-Objects | Sonata |
|------|----------------|--------|
| `torch-scatter` | - | 需要 cu124 版本 |
| `spconv` | - | spconv-cu124 |
| `flash-attention` | 已安装 | 需要 cu124 版本 |
| `torch-scatter` | - | 点云 scatter 操作 |
| `open3d` | 已安装 | 可视化 |
| `timm` | 已安装 | Transformer 模型 |

---

## 2. 安装方案

根据你的需求选择合适的方案：

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| A: 现有环境安装 | 无需切换环境，共享已安装包 | 可能存在 CUDA 版本冲突 | 需要在同一代码中同时使用两个项目 |
| B: 独立环境 | 完全隔离，无冲突 | 需要切换环境 | 推荐方案，稳定可靠 |

---

## 3. 方案A: 在现有环境中安装

> ⚠️ 注意：此方案需要调整 CUDA 版本映射，可能存在兼容性风险

### 3.1 安装 Sonata 核心依赖

```bash
# 激活 sam3d-objects 环境
conda activate sam3d-objects

# 安装 conda 包 (大部分已安装)
conda install -y scipy addict timm psutil huggingface_hub -c conda-forge

# 安装 pip 包 - 调整 CUDA 版本为 cu121
pip install torch-scatter \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html

# 安装 spconv (使用 cu121 版本)
pip install spconv-cu121

# flash-attention 可能需要重新编译
pip install flash-attn --no-build-isolation
```

### 3.2 安装 Sonata 包

```bash
cd sonata
pip install -e .
```

### 3.3 CUDA 版本映射说明

Sonata 原配置使用 CUDA 12.4，需要调整为 CUDA 12.1：

| 原始依赖 | 调整后 |
|----------|--------|
| `torch-scatter ...cu124` | `torch-scatter ...cu121` |
| `spconv-cu124` | `spconv-cu121` |
| `pytorch-cuda=12.4` | 使用现有 `pytorch-cuda=12.1` |

---

## 4. 方案B: 创建独立环境 (推荐)

> ✅ 推荐方案：完全隔离，避免依赖冲突

### 4.1 创建新的 conda 环境

```bash
# 方法1: 使用 Sonata 提供的环境文件
cd sonata
conda env create -f environment.yml
conda activate sonata

# 方法2: 手动创建 (如果想要更精细控制)
conda create -n sonata python=3.10 -y
conda activate sonata

# 安装 PyTorch 和 CUDA
conda install pytorch=2.5.0 torchvision=0.20.0 torchaudio=2.5.0 \
    pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 安装编译工具
conda install gcc=13.2 gxx=13.2 ninja -c conda-forge -y

# 安装科学计算包
conda install numpy scipy addict timm psutil huggingface_hub \
    matplotlib open3d -c conda-forge -y
```

### 4.2 安装 pip 依赖

```bash
# torch-scatter
pip install torch-scatter \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html

# flash-attention (需要编译，耗时较长)
pip install git+https://github.com/Dao-AILab/flash-attention.git

# spconv
pip install spconv-cu124
```

### 4.3 安装 Sonata 包

```bash
cd sonata
pip install -e .
```

---

## 5. 验证安装

### 5.1 验证基础依赖

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# 验证 torch-scatter
import torch_scatter
print("torch-scatter: OK")

# 验证 spconv
import spconv.pytorch as spconv
print("spconv: OK")

# 验证 flash-attention
import flash_attn
print("flash-attn: OK")
```

### 5.2 验证 Sonata

```python
import sonata
print("Sonata: OK")

# 运行示例 (如果下载了预训练模型)
# python sonata/demo/0_pca.py
```

---

## 6. 常见问题

### Q1: torch-scatter 安装失败

**问题**: CUDA 版本不匹配导致安装失败

**解决方案**:
```bash
# 检查当前 CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 根据输出选择正确的 wheel
# CUDA 12.1: torch-2.5.0+cu121
# CUDA 12.4: torch-2.5.0+cu124
pip install torch-scatter \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu${YOUR_CUDA_VERSION}.html
```

### Q2: flash-attention 编译失败

**问题**: 缺少编译依赖或 CUDA 版本不匹配

**解决方案**:
```bash
# 安装编译依赖
conda install gcc gxx ninja -c conda-forge

# 设置 CUDA 路径
export CUDA_HOME=$CONDA_PREFIX

# 重新安装
pip install flash-attn --no-build-isolation
```

### Q3: spconv 导入失败

**问题**: CUDA 版本不匹配

**解决方案**:
```bash
# 卸载错误版本
pip uninstall spconv-cu124 spconv-cu121 -y

# 根据你的 CUDA 版本安装
# CUDA 12.1
pip install spconv-cu121

# CUDA 12.4
pip install spconv-cu124
```

### Q4: 如何在两个环境间切换

```bash
# 使用 SAM-3D-Objects
conda activate sam3d-objects

# 使用 Sonata
conda activate sonata
```

### Q5: 如何同时使用两个项目

如果需要在同一代码中同时使用两个项目，建议：

1. **方案A**: 在 sam3d-objects 环境中安装 Sonata（可能需要调整 CUDA 版本）
2. **进程间通信**: 使用 subprocess 或 RPC 调用另一个环境的 Python
3. **Docker 容器**: 将两个项目分别打包为容器，通过网络通信

---

## 附录: 完整安装命令汇总

### 方案A 一键安装 (sam3d-objects 环境中)

```bash
conda activate sam3d-objects

# 安装缺失的 conda 包
conda install -y scipy addict timm psutil huggingface_hub -c conda-forge

# 安装 pip 包 (CUDA 12.1 版本)
pip install torch-scatter \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install spconv-cu121
pip install flash-attn --no-build-isolation

# 安装 Sonata
cd sonata && pip install -e .
```

### 方案B 一键安装 (独立环境)

```bash
# 创建环境
conda env create -f sonata/environment.yml
conda activate sonata

# 安装 Sonata
cd sonata && pip install -e .
```

---

*文档生成日期: 2026-03-10*
