# 3D 点云语义分割与重建数据集综述

> 本文档整理了与 SAM-3D-Objects、Sonata 和点云语义分割相关的数据集信息，为 `train_semantic_segmentation.py` 提供数据集选择参考。

---

## 目录

1. [项目数据集需求分析](#1-项目数据集需求分析)
2. [室内 3D 语义分割数据集](#2-室内-3d-语义分割数据集)
3. [室外 3D 语义分割数据集](#3-室外-3d-语义分割数据集)
4. [单图像 3D 重建数据集](#4-单图像-3d-重建数据集)
5. [3D 预训练数据集](#5-3d-预训练数据集)
6. [Sonata 预训练数据集](#6-sonata-预训练数据集)
7. [数据集对比与推荐](#7-数据集对比与推荐)
8. [数据集获取指南](#8-数据集获取指南)
9. [参考文献](#9-参考文献)

---

## 1. 项目数据集需求分析

### 1.1 train_semantic_segmentation.py 数据格式要求

根据 `sam3d_objects/semantic/dataset.py` 的定义，项目需要的数据格式：

```
数据集根目录/
├── train.txt                    # 训练场景列表（可选）
├── val.txt                      # 验证场景列表（可选）
├── scene_001/                    # 场景目录
│   ├── point_cloud.npy          # (N, 6) xyz坐标 + rgb颜色
│   ├── image.png                 # RGB图像
│   ├── mask.png                  # 2D对象掩码
│   ├── label.npy                 # (N,) 点云二进制标签
│   └── intrinsics.json           # 相机内参（可选）
├── scene_002/
│   └── ...
```

### 1.2 核心数据组成

| 数据项 | 形状 | 说明 |
|--------|------|------|
| point_cloud | (N, 6) | xyz坐标 + rgb颜色 |
| image | (H, W, 3) | RGB图像 |
| mask | (H, W) | 2D对象掩码 |
| label | (N,) | 点云二进制标签 |
| intrinsics | (3, 3) | 相机内参矩阵 |

### 1.3 模型输入输出

**输入**:
- 点云场景 (N, 3) - 已有的 3D 点云
- 单视图 RGB 图像 (3, H, W)
- 单视图对应的 2D 物体掩码 (1, H, W)
- 相机内参 (fx, fy, cx, cy)

**输出**:
- 点云场景中物体的掩码 (N,) 或 (N, C)
- 每个点是否属于目标物体

---

## 2. 室内 3D 语义分割数据集

### 2.1 ScanNet

**简介**: 大规模室内 RGB-D 数据集，包含 1,500+ 场景扫描，2.5M 视图。

| 属性 | 值 |
|------|-----|
| 场景数 | 1,513 场景 |
| 视图数 | 2.5M+ RGB-D 帧 |
| 类别数 | 20 (ScanNet20) / 200 (ScanNet200) |
| 数据类型 | RGB-D 视频、3D 网格、语义/实例标注 |
| 官网 | http://www.scan-net.org/ |
| 下载 | 需申请访问权限 |

**特点**:
- 完整的相机位姿和表面重建
- 实例级语义分割标注
- 支持 3D 物体检测、语义分割、场景理解等任务

**ScanNet++ (2023)**:
- 更高质量：33MP DSLR 图像、亚毫米激光扫描
- 更大规模：460+ 场景
- 官网：https://scannetpp.mlsg.cit.tum.de/

**适用性**: ⭐⭐⭐⭐⭐ 非常适合本项目的室内场景训练

### 2.2 S3DIS (Stanford Large-Scale 3D Indoor Spaces)

**简介**: 斯坦福大规模室内 3D 场景数据集。

| 属性 | 值 |
|------|-----|
| 区域数 | 6 个大型室内区域 |
| 房间数 | 271 个房间 |
| 类别数 | 13 个语义类别 |
| 点数 | 约 6.9 亿点 |
| 数据类型 | 彩色点云、语义/实例标注 |
| 官网 | http://buildingparser.stanford.edu/dataset.html |
| 下载 | 需申请 |

**类别列表**:
ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board, clutter

**适用性**: ⭐⭐⭐⭐⭐ 非常适合本项目的室内场景训练

### 2.3 Matterport3D

**简介**: 大规模 RGB-D 室内场景数据集。

| 属性 | 值 |
|------|-----|
| 场景数 | 90 个建筑级场景 |
| 视图数 | 10,800 全景视图，194,400 RGB-D 图像 |
| 数据类型 | RGB-D、3D 网格、语义分割 |
| 官网 | https://niessner.github.io/Matterport/ |
| 下载 | 需申请访问权限 |

**特点**:
- 建筑规模的场景
- 实例级语义分割
- 支持导航、场景理解等任务

### 2.4 HM3D (Habitat-Matterport 3D)

**简介**: 最大的 3D 室内空间数据集。

| 属性 | 值 |
|------|-----|
| 场景数 | 1,000 个高质量 3D 扫描 |
| 类型 | 住宅和商业建筑 |
| 导航面积 | 1.4-3.7x 大于之前数据集 |
| 官网 | https://aihabitat.org/datasets/hm3d/ |

**HM3D-Semantics**:
- 密集语义标注
- 最大规模的室内语义数据集
- 支持 embodied AI 任务

### 2.5 ARKitScenes

**简介**: Apple 发布的多样化室内 3D 场景数据集。

| 属性 | 值 |
|------|-----|
| 场景数 | 5,047 个场景 |
| 数据类型 | RGB-D、激光扫描、3D 边界框 |
| 特点 | 首个使用广泛可用深度传感器采集的数据集 |
| 官网 | https://machinelearning.apple.com/research/arkitscenes |
| 下载 | GitHub: https://github.com/apple/ARKitScenes |

**ARKit LabelMaker (CVPR 2025)**:
- 比之前最大数据集大 3 倍
- 密集语义标注

### 2.6 ScanNet200

**简介**: ScanNet 的扩展版本，支持 200 类语义分割。

| 属性 | 值 |
|------|-----|
| 类别数 | 200 类 |
| 特点 | 比之前 3D 场景数据集多一个数量级的类别 |
| 官网 | https://rozdavid.github.io/scannet200 |

---

## 3. 室外 3D 语义分割数据集

### 3.1 SemanticKITTI

**简介**: 基于 KITTI 的大规模 LiDAR 语义分割数据集。

| 属性 | 值 |
|------|-----|
| 序列数 | 22 个序列（10 个训练，11 个测试） |
| 点数 | 约 43,000 帧扫描 |
| 类别数 | 19/20 类 |
| 数据类型 | LiDAR 点云、语义标注 |
| 官网 | http://semantic-kitti.org/ |
| 下载 | 开放下载 |

**特点**:
- 自动驾驶场景
- 完整的序列标注
- 支持 3D 语义场景补全

**SemanticKITTI-C**:
- 16 种域外损坏类型
- 恶劣天气、测量噪声等

### 3.2 nuScenes

**简介**: 大规模自动驾驶数据集。

| 属性 | 值 |
|------|-----|
| 场景数 | 1,000 个场景 |
| 帧数 | 1.4M 帧 |
| 传感器 | 1x LiDAR, 5x RADAR, 6x 相机 |
| 类别数 | 23 类（语义分割） |
| 官网 | https://www.nuscenes.org/ |
| 下载 | 开放下载 |

**特点**:
- 完整的传感器套件
- 3D 物体检测和语义分割
- 时间序列数据

### 3.3 Waymo Open Dataset

**简介**: Google Waymo 发布的大规模自动驾驶数据集。

| 属性 | 值 |
|------|-----|
| 场景数 | 1,150 个场景 |
| 传感器 | 5x LiDAR, 5x 相机 |
| 数据类型 | LiDAR 点云、相机图像、标注 |
| 官网 | https://waymo.com/open/ |

### 3.4 Semantic3D

**简介**: 大规模室外点云语义分割数据集。

| 属性 | 值 |
|------|-----|
| 点数 | 超过 40 亿点 |
| 场景 | 城市场景 |
| 类别数 | 8 类 |
| 官网 | http://semantic3d.net/ |
| 下载 | 开放下载 |

**类别**: man-made terrain, natural terrain, high vegetation, low vegetation, buildings, hard scape, scanning artefacts, cars

---

## 4. 单图像 3D 重建数据集

### 4.1 CO3D (Common Objects in 3D)

**简介**: Meta 发布的多视角 3D 物体数据集。

| 属性 | 值 |
|------|-----|
| 视频数 | 约 19,000 个视频 |
| 帧数 | 1.5M 帧 |
| 类别数 | 50+ 类 |
| 数据类型 | 多视角图像、深度图、掩码、相机位姿、3D 点云 |
| 官网 | https://ai.meta.com/datasets/co3d-dataset/ |
| 下载 | 开放下载 |

**特点**:
- 真实世界多视角图像
- 类别特定的 3D 重建
- 新视角合成

**uCO3D (Uncommon Objects in 3D)**:
- 170k 视频，~1k 类别
- 19.3TB 数据
- 最大规模的真实世界 3D 物体数据集

### 4.2 Objaverse

**简介**: 大规模 3D 物体数据集。

| 属性 | 值 |
|------|-----|
| 物体数 | 800K+ 3D 物体 |
| 类型 | 多样化的 3D 模型 |
| 官网 | https://objaverse.allenai.org/ |
| 下载 | HuggingFace |

**Objaverse-XL**:
- 10M+ 3D 物体
- 多样化来源
- 用于训练 Zero123-XL

### 4.3 MVImgNet

**简介**: 大规模多视角图像数据集。

| 属性 | 值 |
|------|-----|
| 物体数 | ~220K 物体 |
| 类别数 | 238 类 |
| 数据类型 | 多视角图像、相机位姿、3D 点云 |
| 论文 | CVPR 2023 |

**MVImgNet2.0**:
- 更大规模
- 更高质量

**MVPNet**:
- 从 MVImgNet 派生的 3D 点云数据集
- 87,200 点云，150 类别

### 4.4 ShapeNet

**简介**: 大规模 3D CAD 模型数据集。

| 属性 | 值 |
|------|-----|
| 模型数 | 3M+ CAD 模型 |
| 类别数 | 3,135 WordNet synsets |
| 子集 | ShapeNetCore (51K), ShapeNetSem (12K) |
| 官网 | https://shapenet.org/ |
| 下载 | 开放下载 |

**用途**:
- 3D 分类和分割
- 单图像 3D 重建
- 形状生成

### 4.5 ModelNet

**简介**: 普林斯顿 3D 物体分类数据集。

| 属性 | 值 |
|------|-----|
| 模型数 | 127,915 个 CAD 模型 |
| 类别数 | 662 类（ModelNet40: 40 类） |
| 官网 | https://modelnet.cs.princeton.edu/ |
| 下载 | Kaggle, GitHub |

**ModelNet40-C**:
- 15 种损坏类型
- 用于评估鲁棒性

### 4.6 PartNet

**简介**: 大规模 3D 部件分割数据集。

| 属性 | 值 |
|------|-----|
| 模型数 | 26,671 个 3D 模型 |
| 部件数 | 573,585 个部件实例 |
| 类别数 | 24 类 |
| 官网 | https://partnet.cs.stanford.edu/ |
| 下载 | GitHub |

**PartNeXt (2025)**:
- 扩展到 50 类
- 23,000+ 高质量纹理模型

---

## 5. 3D 预训练数据集

### 5.1 ULIP 数据集

**ULIP-Objaverse**:
- 点云、图像、语言描述三元组
- 基于 Objaverse 构建

**ULIP-ShapeNet**:
- 点云、图像、语言描述三元组
- 基于 ShapeNet 构建

**官网**: https://github.com/salesforce/ULIP

### 5.2 OpenShape 数据集

**简介**: 大规模 3D 形状表示学习数据集。

| 数据来源 | 说明 |
|----------|------|
| ShapeNetCore | 51K 模型 |
| 3D-FUTURE | 20K 模型 |
| Objaverse | 800K 模型 |
| LVIS 类别 | 对齐视觉概念 |

**官网**: https://github.com/Colin97/OpenShape_code

### 5.3 Point-E / Zero-1-to-3 数据

**Zero-1-to-3**:
- 基于 Objaverse 训练
- 学习相机视角变化

**Point-E**:
- 文本到 3D 生成
- 使用 Objaverse 和其他数据集

---

## 6. Sonata 预训练数据集

### 6.1 Sonata 预训练数据

根据 Sonata 论文 (CVPR 2025)，预训练使用：

| 数据集 | 规模 | 说明 |
|--------|------|------|
| 140K 点云 | 140,000 | 自蒸馏预训练数据 |
| ScanNet | 1,513 场景 | 下游评估 |
| S3DIS | 6 区域 | 下游评估 |
| SemanticKITTI | 22 序列 | 下游评估 |
| nuScenes | 1,000 场景 | 下游评估 |
| HM3D | 1,000 场景 | 预训练 |
| ARKitScenes | 5,047 场景 | 预训练 |

### 6.2 Pointcept 支持的数据集

Pointcept 框架支持预处理：

**室内数据集**:
- ScanNet20 / ScanNet200 / ScanNet Data Efficient
- S3DIS
- Matterport3D
- ARKitScenes

**室外数据集**:
- SemanticKITTI
- nuScenes
- Waymo Open Dataset
- Semantic3D

**预处理脚本**:
```bash
# ScanNet 预处理
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py

# S3DIS 预处理
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py

# SemanticKITTI 预处理
python pointcept/datasets/preprocessing/semantic_kitti/preprocess_semantickitti.py
```

---

## 7. 数据集对比与推荐

### 7.1 按任务推荐

| 任务 | 推荐数据集 | 原因 |
|------|-----------|------|
| 室内语义分割 | ScanNet, S3DIS | 大规模、高质量标注 |
| 室外语义分割 | SemanticKITTI, nuScenes | 自动驾驶场景 |
| 单图像 3D 重建 | CO3D, Objaverse | 多视角、多样物体 |
| 3D 预训练 | Objaverse-XL, HM3D | 规模大、多样性 |
| 部件分割 | PartNet, ShapeNetPart | 细粒度部件标注 |

### 7.2 按数据规模对比

| 数据集 | 场景/物体数 | 点数/帧数 | 类别数 |
|--------|------------|----------|--------|
| ScanNet | 1,513 | 2.5M | 20/200 |
| S3DIS | 271 房间 | 690M | 13 |
| SemanticKITTI | 22 序列 | 43K 帧 | 19 |
| nuScenes | 1,000 | 1.4M | 23 |
| Objaverse | 800K | - | - |
| CO3D | 19K 视频 | 1.5M | 50+ |

### 7.3 项目推荐数据集组合

针对 `train_semantic_segmentation.py` 的训练需求：

**方案 A: 室内场景**
```
训练数据: ScanNet (训练集)
验证数据: ScanNet (验证集) / S3DIS (交叉验证)
测试数据: S3DIS Area 5 / ScanNet 测试集
```

**方案 B: 室外场景**
```
训练数据: SemanticKITTI (序列 00-10)
验证数据: SemanticKITTI (序列 08)
测试数据: SemanticKITTI (序列 11-21)
```

**方案 C: 混合预训练**
```
预训练: HM3D + ARKitScenes + ScanNet
微调: S3DIS / SemanticKITTI
```

---

## 8. 数据集获取指南

### 8.1 开放下载数据集

| 数据集 | 下载方式 | 许可证 |
|--------|----------|--------|
| SemanticKITTI | http://semantic-kitti.org/ | CC BY-NC-SA 4.0 |
| nuScenes | https://www.nuscenes.org/ | CC BY-NC-SA 4.0 |
| ModelNet40 | Kaggle, GitHub | 开放 |
| ShapeNet | https://shapenet.org/ | 开放 |
| Objaverse | HuggingFace | 开放 |
| CO3D | https://ai.meta.com/datasets/co3d-dataset/ | CC BY-NC 4.0 |
| PartNet | GitHub | 开放 |

### 8.2 需申请数据集

| 数据集 | 申请方式 | 审核时间 |
|--------|----------|----------|
| ScanNet | http://www.scan-net.org/ | 1-2 周 |
| S3DIS | Stanford 官网 | 1-2 周 |
| Matterport3D | https://niessner.github.io/Matterport/ | 1-2 周 |
| HM3D | https://aihabitat.org/datasets/hm3d/ | 1-2 周 |
| ARKitScenes | GitHub | 即时 |
| Waymo | https://waymo.com/open/ | 即时 |

### 8.3 数据集预处理工具

**Pointcept 预处理**:
```bash
# 安装 Pointcept
pip install pointcept

# 预处理 ScanNet
python -m pointcept.datasets.preprocessing.scannet.preprocess_scannet

# 预处理 S3DIS
python -m pointcept.datasets.preprocessing.s3dis.preprocess_s3dis
```

**MMDetection3D 预处理**:
```bash
pip install mmdet3d

# ScanNet 预处理
python tools/data_converter/scannet_data_utils.py

# SemanticKITTI 预处理
python tools/data_converter/semantickitti_converter.py
```

### 8.4 数据格式转换

将标准数据集转换为项目格式：

```python
import numpy as np
from PIL import Image
import json

def convert_scannet_to_project_format(scannet_dir, output_dir, scene_id):
    """将 ScanNet 转换为项目数据格式"""
    
    # 加载 ScanNet 数据
    # ... 
    
    # 保存为项目格式
    np.save(f"{output_dir}/{scene_id}/point_cloud.npy", point_cloud)
    Image.fromarray(image).save(f"{output_dir}/{scene_id}/image.png")
    Image.fromarray(mask).save(f"{output_dir}/{scene_id}/mask.png")
    np.save(f"{output_dir}/{scene_id}/label.npy", labels)
    
    # 相机内参
    intrinsics = {
        "fx": 500.0, "fy": 500.0,
        "cx": 320.0, "cy": 240.0
    }
    with open(f"{output_dir}/{scene_id}/intrinsics.json", "w") as f:
        json.dump(intrinsics, f)
```

---

## 9. 参考文献

### 3D 语义分割数据集论文

1. **ScanNet**: Dai, A., et al. "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes." CVPR 2017.

2. **S3DIS**: Armeni, I., et al. "3D Semantic Parsing of Large-Scale Indoor Spaces." CVPR 2016.

3. **SemanticKITTI**: Behley, J., et al. "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences." ICCV 2019.

4. **nuScenes**: Caesar, H., et al. "nuScenes: A Multimodal Dataset for Autonomous Driving." CVPR 2020.

5. **Matterport3D**: Chang, A., et al. "Matterport3D: Learning from RGB-D Data in Indoor Environments." 3DV 2017.

6. **HM3D**: Ramakrishnan, S. K., et al. "Habitat-Matterport 3D Dataset (HM3D)." NeurIPS 2021.

7. **ARKitScenes**: Dehghan, M., et al. "ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene Understanding." ICCV 2021.

### 3D 重建数据集论文

8. **CO3D**: Reizenstein, J., et al. "Common Objects in 3D: Large-Scale Learning and Evaluation of Real-World 3D Categorization." ICCV 2021.

9. **Objaverse**: Deitke, M., et al. "Objaverse: A Universe of Annotated 3D Objects." CVPR 2023.

10. **MVImgNet**: Yu, X., et al. "MVImgNet: A Large-Scale Dataset of Multi-View Images." CVPR 2023.

11. **ShapeNet**: Chang, A. X., et al. "ShapeNet: An Information-Rich 3D Model Repository." arXiv 2015.

12. **PartNet**: Mo, K., et al. "PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding." CVPR 2019.

### 相关方法论文

13. **Sonata**: Wu, X., et al. "Sonata: Self-Supervised Learning of Reliable Point Representations." CVPR 2025.

14. **PTv3**: Wu, X., et al. "Point Transformer V3: Simpler, Faster, Stronger." CVPR 2024.

15. **SAM 3D**: "SAM 3D: 3Dfy Anything in Images." arXiv 2025.

16. **OpenShape**: Liu, Z., et al. "OpenShape: Scaling Up 3D Shape Representation Towards Open-World Recognition." NeurIPS 2023.

17. **ULIP**: Xue, L., et al. "ULIP: Learning Unified Representation of Language, Images, and Point Clouds." CVPR 2023.

18. **OpenScene**: Peng, S., et al. "OpenScene: 3D Scene Understanding with Open Vocabularies." CVPR 2023.

19. **Mask3D**: Schult, J., et al. "Mask3D: Mask Transformer for 3D Semantic Instance Segmentation." ICRA 2023.

20. **SAM3D**: Yang, Z., et al. "SAM3D: Segment Anything in 3D Scenes." ICCV 2023 Workshop.

---

## 10. 补充数据集

### 10.1 OmniObject3D

**简介**: 大词汇量真实世界 3D 物体数据集。

| 属性 | 值 |
|------|-----|
| 物体数 | 6,000 个专业扫描物体 |
| 类别数 | 190 个日常类别 |
| 数据类型 | 纹理网格、点云、多视角渲染图像、真实捕获视频 |
| 官网 | https://omniobject3d.github.io/ |
| 下载 | OpenDataLab, GitHub |

**特点**:
- CVPR 2023 Award Candidate
- 真实世界扫描物体
- 支持 3D 感知、重建和生成

### 10.2 3D-FRONT

**简介**: 大规模合成室内场景数据集。

| 属性 | 值 |
|------|-----|
| 房屋数 | 6,813 套房屋 |
| 房间数 | 18,968 个房间 |
| 家具数 | 13,151 个独特家具 |
| 数据类型 | 房屋布局、家具模型、纹理 |
| 下载 | GitHub, 阿里云天池 |

**特点**:
- 专业设计师设计
- 高质量家具模型
- 支持室内场景合成和纹理合成

### 10.3 NYU Depth V2

**简介**: 经典的室内 RGB-D 数据集。

| 属性 | 值 |
|------|-----|
| 帧数 | 1,449 幅标注的 RGB-D 图像 |
| 视频 | 407,024 帧视频序列 |
| 类别数 | 894 类（40 类常用） |
| 场景 | 商业和住宅建筑 |
| 官网 | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html |
| 下载 | Kaggle, HuggingFace, TensorFlow Datasets |

**特点**:
- 经典的室内场景理解数据集
- 支持 2D/3D 语义分割
- 深度估计基准

### 10.4 ScanRefer / ReferIt3D

**简介**: 3D 视觉定位数据集。

**ScanRefer**:
| 属性 | 值 |
|------|-----|
| 描述数 | 51,583 条自然语言描述 |
| 场景 | ScanNet 场景 |
| 任务 | 3D 物体定位、视觉定位 |
| 下载 | GitHub (需申请) |

**ReferIt3D**:
- 基于 ScanNet 的 3D 引用表达式理解
- 细粒度物体类识别
- 支持 3D 视觉对话

### 10.5 Mip-NeRF 360 / Tanks and Temples

**简介**: 3D 场景重建基准数据集。

**Mip-NeRF 360**:
| 属性 | 值 |
|------|-----|
| 场景数 | 9 个大规模真实世界场景 |
| 类型 | 室内外无界场景 |
| 用途 | NeRF/Gaussian Splatting 评估 |

**Tanks and Temples**:
| 属性 | 值 |
|------|-----|
| 场景数 | 21 个场景 |
| 类型 | 室内外大尺度场景 |
| 特点 | 实验室外采集，真实条件 |
| 用途 | 3D 重建基准 |

---

## 11. 数据集使用示例

### 11.1 ScanNet 数据准备

```python
# ScanNet 数据下载和预处理
import os
import numpy as np

# 1. 申请 ScanNet 访问权限
# 访问 http://www.scan-net.org/ 填写申请表

# 2. 下载数据 (使用 ScanNet 下载脚本)
# python reader.py --filename scene0000_00.sens --output_path ./

# 3. 预处理为项目格式
def preprocess_scannet_scene(scene_dir, output_dir):
    """将 ScanNet 场景转换为项目格式"""
    import open3d as o3d
    from PIL import Image
    
    # 加载 ScanNet 数据
    mesh = o3d.io.read_triangle_mesh(f"{scene_dir}/mesh.ply")
    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    
    # 提取坐标和颜色
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    
    # 合并为 point_cloud.npy
    point_cloud = np.concatenate([points, colors], axis=1)
    np.save(f"{output_dir}/point_cloud.npy", point_cloud)
    
    # 加载图像和掩码
    frame_id = 0  # 选择关键帧
    image = Image.open(f"{scene_dir}/frame-{frame_id:06d}.color.jpg")
    depth = Image.open(f"{scene_dir}/frame-{frame_id:06d}.depth.png")
    
    # 保存
    image.save(f"{output_dir}/image.png")
    
    print(f"预处理完成: {output_dir}")

# 4. 使用 Pointcept 预处理
# python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py \
#     --dataset_root /path/to/scannet \
#     --output_root /path/to/processed
```

### 11.2 SemanticKITTI 数据准备

```python
# SemanticKITTI 数据下载和预处理

# 1. 下载数据集
# 访问 http://semantic-kitti.org/ 下载

# 2. 数据结构
# semantic-kitti/
# ├── sequences/
# │   ├── 00/
# │   │   ├── velodyne/
# │   │   ├── labels/
# │   │   └── calib.txt
# │   ├── 01/
# │   └── ...

# 3. 预处理脚本
def preprocess_semantickitti(sequence_dir, output_dir, frame_id):
    """将 SemanticKITTI 帧转换为项目格式"""
    import numpy as np
    from PIL import Image
    
    # 加载点云
    points = np.fromfile(
        f"{sequence_dir}/velodyne/{frame_id:06d}.bin",
        dtype=np.float32
    ).reshape(-1, 4)[:, :3]  # 只取 xyz
    
    # 加载标签
    labels = np.fromfile(
        f"{sequence_dir}/labels/{frame_id:06d}.label",
        dtype=np.uint32
    )
    # 提取语义标签
    semantic_labels = labels & 0xFFFF
    
    # 加载相机参数
    calib = np.loadtxt(f"{sequence_dir}/calib.txt")
    
    # 保存
    np.save(f"{output_dir}/point_cloud.npy", 
            np.concatenate([points, np.zeros((len(points), 3))], axis=1))
    np.save(f"{output_dir}/label.npy", semantic_labels)
    
    print(f"预处理完成: {output_dir}")
```

### 11.3 CO3D 数据准备

```python
# CO3D 数据下载和预处理

# 1. 下载数据集
# pip install co3d
# from co3d.dataset.data_types import load_dataclass_jgzip

# 2. 数据结构
# co3d/
# ├── apple/
# │   ├── frame_annotations.jgz
# │   ├── set_lists/
# │   └── images/
# ├── ball/
# └── ...

# 3. 使用官方工具
from co3d.dataset.data_types import load_dataclass_jgzip
from co3d.dataset.dataset_zoo import dataset_zoo

# 加载数据集
datasets = dataset_zoo(
    category="apple",
    dataset_root="/path/to/co3d",
)

# 获取训练数据
train_dataset = datasets["train"]
for idx in range(len(train_dataset)):
    frame_data = train_dataset[idx]
    # frame_data.image: RGB 图像
    # frame_data.depth: 深度图
    # frame_data.mask: 物体掩码
    # frame_data.camera: 相机参数
```

---

## 12. 数据集统计信息

### 12.1 室内数据集统计

| 数据集 | 场景数 | 点数 | 类别数 | 标注类型 |
|--------|--------|------|--------|----------|
| ScanNet | 1,513 | 2.5M 帧 | 20/200 | 语义+实例 |
| S3DIS | 271 房间 | 690M | 13 | 语义+实例 |
| Matterport3D | 90 | 194K 帧 | 40 | 语义+实例 |
| HM3D | 1,000 | - | - | 导航+语义 |
| ARKitScenes | 5,047 | - | - | 检测+分割 |
| NYUv2 | 464 场景 | 407K 帧 | 894 | 语义+深度 |

### 12.2 室外数据集统计

| 数据集 | 序列数 | 帧数 | 类别数 | 传感器 |
|--------|--------|------|--------|--------|
| SemanticKITTI | 22 | 43K | 19 | LiDAR |
| nuScenes | 1,000 | 1.4M | 23 | LiDAR+相机 |
| Waymo | 1,150 | - | - | LiDAR+相机 |
| Semantic3D | 30 | 4B 点 | 8 | 激光扫描 |

### 12.3 物体数据集统计

| 数据集 | 物体数 | 类别数 | 数据类型 |
|--------|--------|--------|----------|
| Objaverse | 800K | - | 3D 模型 |
| CO3D | 19K 视频 | 50+ | 多视角图像 |
| ShapeNet | 3M | 3,135 | CAD 模型 |
| ModelNet | 127K | 662 | CAD 模型 |
| PartNet | 26K | 24 | 部件分割 |
| OmniObject3D | 6K | 190 | 扫描模型 |

---

## 13. 常见问题与解决方案

### Q1: 如何选择合适的数据集？

**建议**:
- 室内语义分割: ScanNet (大规模) 或 S3DIS (标准基准)
- 室外语义分割: SemanticKITTI (标准基准) 或 nuScenes (多模态)
- 单图像 3D 重建: CO3D (真实多视角) 或 Objaverse (大规模合成)
- 预训练: HM3D + ARKitScenes (室内) 或 Objaverse-XL (物体)

### Q2: 数据集下载速度慢怎么办？

**解决方案**:
- 使用镜像站点 (如 OpenDataLab)
- 使用代理或加速器
- 分批次下载

### Q3: 如何处理不同数据集的格式差异？

**解决方案**:
- 使用 Pointcept 或 MMDetection3D 的预处理工具
- 编写格式转换脚本
- 统一使用项目定义的数据格式

### Q4: 内存不足如何处理大规模数据集？

**解决方案**:
- 使用数据流式加载
- 分块处理点云
- 使用稀疏卷积

---

*文档更新日期: 2026-03-10*
