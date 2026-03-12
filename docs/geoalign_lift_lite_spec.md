# GeoAlign-Lift-Lite 实现规格

## 1. 目标

本规格文档将 `GeoAlign-Lift-Lite` 从方法想法收紧为可直接实现的训练原型。

最终目标：

- 输入：`image + 2D mask + scene 3DGS gaussians`
- 输出：`gaussian_mask_logits / gaussian_mask_probs`
- 用途：
  - 直接作为 3D 物体分割结果
  - 或作为 `Semantic Gaussians` 中 feature fusion 的前景筛选器

第一版以“单视角、轻量级、可训练稳定”为原则，不追求最强性能。

## 2. 第一版系统边界

### 2.1 要做的

- 训练一个条件高斯掩码预测模型
- 支持高斯级二值监督
- 支持弱监督几何一致性与 2D 重投影
- 支持接入 SAM3D 几何先验
- 支持后续接入 `semantic-gaussians/fusion.py`

### 2.2 不做的

- 不做多视角联合编码
- 不做开放词汇类别识别
- 不端到端训练 DINOv2 / Sonata / SAM3D
- 不做对象级 memory bank / codebook
- 不做全场景 panoptic segmentation

## 3. 模块拆分

建议将第一版实现拆成 5 个模块。

### 3.1 `dataset`

负责组织单视角训练样本。

每个样本字段建议固定为：

- `image`: `(3, H, W)`
- `mask_2d`: `(1, H, W)`
- `gaussian_xyz`: `(N, 3)`
- `gaussian_attr`: `(N, Dg)`
- `camera_intrinsics`: `(3, 3)`
- `camera_extrinsics`: `(4, 4)` 或等效表示
- `gaussian_label`: `(N,)`，可选
- `scene_name`
- `view_name`

### 3.2 `geometry_prior_extractor`

负责从 `(image, mask_2d)` 提取 SAM3D 几何先验。

输出建议固定为：

- `shape_latent`: `(C_geo, D, H, W)`，默认 `8 x 16 x 16 x 16`
- `pointmap`: `(H_p, W_p, 3)` 或等效张量
- `occupancy_meta`
- `surface_meta`

第一版可允许离线缓存，避免训练时重复跑 SAM3D。

### 3.3 `gaussian_feature_builder`

负责将原始高斯属性整理成模型输入特征。

第一版特征建议：

- `xyz`
- `features_dc` 或 RGB
- `opacity`
- `log_scale`
- `rotation`

构成：

- `gaussian_attr: (N, Dg)`

若加入 Sonata，则再拼接一个低维投影后的 `sonata_feat`。

### 3.4 `geoalign_lift_lite_model`

负责前向推理。

输入：

- `image`
- `mask_2d`
- `gaussian_xyz`
- `gaussian_attr`
- `shape_latent`
- `pointmap`

输出：

- `mask_logits: (N,)`
- `mask_probs: (N,)`
- `aux_dict`

### 3.5 `loss_and_renderer`

负责：

- 高斯级分割损失
- 2D 重投影损失
- 几何一致性损失

如果第一版渲染高斯掩码成本过高，可先实现简化投影版本，而不是完整高斯 rasterization。

## 4. 输入特征规格

## 4.1 2D 分支输入

第一版固定使用：

- 原始 RGB 图像
- 二值目标 mask

2D encoder 默认实现：

- 冻结 DINOv2
- 提取 patch tokens
- 用 mask pooling 得到单个 `object token`

输出：

- `t_obj: (C,)`

建议维度：

- 若 DINO 输出 768 维，则用一个线性层压到 `128`

## 4.2 高斯分支输入

每个高斯建议构造如下原始特征：

- `x, y, z`
- `r, g, b`
- `opacity`
- `sx, sy, sz` 使用 log-scale
- `rotation` 的 4 维四元数或紧凑表示

推荐维度预算：

- `3 + 3 + 1 + 3 + 4 = 14`

再加位置编码：

- `PE(xyz)`，可选 16 或 32 维

组合后送入一个小 MLP，投到：

- `D = 128`

## 4.3 几何先验特征

从 SAM3D 结果中提取每个高斯的几何特征：

- `occ_score`
- `surface_distance`
- `inside_bbox_flag`
- `pointmap_visibility_score`

第一版建议只做 2 个核心量：

- `shape occupancy score`
- `distance to pointmap surface`

几何特征维度控制在：

- `G = 8` 以内

## 4.4 Sonata 特征接入策略

为了保持轻量，分两阶段：

### 第一阶段

- 不使用 Sonata

### 第二阶段

- 使用冻结 Sonata
- 提取场景点云特征
- 通过高斯中心的最近邻或半径聚合映射到高斯
- 用线性层将 512 维压缩到 64 维

拼接后总特征维度仍控制在 192 左右。

## 5. 网络结构规格

### 5.1 推荐实现

第一版推荐：

1. `2D object token projector`
   - `768 -> 128`
2. `Gaussian feature encoder`
   - 小 MLP，将高斯属性编码到 128 维
3. `Geometry feature encoder`
   - 小 MLP，将几何特征编码到 32 或 64 维
4. `Fusion head`
   - 拼接后维度约 192
   - 用 object token 做 gated modulation 或单层 cross-attention
5. `Mask decoder`
   - 2 层 MLP 输出单个 logit

### 5.2 更轻的默认版本

若优先实现速度，推荐使用：

- `gated MLP`

具体形式：

`z_i = MLP([g_attr_i ; g_geo_i])`

`gate = sigmoid(W t_obj)`

`z_i' = z_i * gate`

`logit_i = MLP(z_i')`

这能避免 transformer 开销。

### 5.3 输出

输出头固定为：

- `mask_logits = (N,)`
- `mask_probs = sigmoid(mask_logits)`

辅助诊断输出建议保留：

- `geo_scores`
- `reproj_mask`

## 6. 标签与伪标签构造

## 6.1 强监督标签

若已有场景物体对应的 3D 高斯标签：

- 直接用作 `gaussian_label`

标签定义：

- `1`: 属于当前目标
- `0`: 不属于当前目标
- 可选忽略区：`-1`

## 6.2 伪标签生成

如果没有高斯级真值，第一版推荐构造高精度伪标签：

1. 用当前相机将高斯投影到 2D。
2. 选取投影到 mask 内的高斯。
3. 用深度一致性筛选。
4. 用 SAM3D `shape occupancy` 和 `pointmap surface distance` 再筛一遍。
5. 将高置信候选作为伪正样本。
6. 将明显在 mask 外且几何不一致的高斯作为伪负样本。

中间区域不强行监督，可作为 ignore。

### 第一版建议

- 只用高精度正负样本
- 不追求全覆盖
- 用模型学习从“高精度但稀疏的伪标签”泛化到完整高斯掩码

## 7. 损失设计

## 7.1 主损失

`L_mask`

建议：

- `BCE + Dice`

理由：

- 前景/背景不平衡明显
- Dice 对小物体更稳

## 7.2 几何一致性损失

`L_geo`

包括：

- `L_occ`
  - 高概率高斯应位于 SAM3D 估计的目标占据区域内
- `L_surface`
  - 高概率高斯不应离目标表面过远

第一版可以写成软约束：

- `prob * distance`
- `prob * (1 - occupancy)`

## 7.3 重投影损失

`L_reproj`

将预测的高斯掩码投影到当前图像平面，得到 soft 2D mask，与输入 `mask_2d` 对齐。

若完整高斯渲染实现复杂，第一版可用简化版本：

- 将高斯中心投影为点
- 在图像平面上生成稀疏 heatmap
- 对 heatmap 进行高斯平滑

然后与 `mask_2d` 计算 BCE / Dice。

## 7.4 稀疏性与面积约束

`L_sparse`

作用：

- 防止模型全选
- 鼓励分割集中

第一版可以简单使用：

- `mean(mask_probs)`

或与 2D mask 面积建立弱比例约束。

## 7.5 总损失

推荐默认配置：

`L = L_mask + 0.2 * L_geo + 0.5 * L_reproj + 0.05 * L_sparse`

若没有强监督：

`L = 0.5 * L_geo + 1.0 * L_reproj + 0.05 * L_sparse`

但需要明确说明这是较弱配置，仅适合预训练或 warm start。

## 8. 训练流程

### 8.1 数据准备

每个场景预先缓存：

- 场景高斯
- 每张图像
- 2D target masks
- 相机参数
- SAM3D geometry prior

### 8.2 训练步骤

1. 读取一个单视角样本
2. 构造高斯属性特征
3. 读取对应 SAM3D 几何先验
4. 提取 object token
5. 预测高斯掩码
6. 计算 `L_mask + L_geo + L_reproj + L_sparse`
7. 更新仅有的小型融合头和解码器

### 8.3 冻结策略

第一版默认：

- 冻结 DINOv2
- 冻结 SAM3D
- 冻结 Sonata

只训练：

- 高斯特征编码器
- 几何特征编码器
- 融合头
- mask decoder

## 9. 推理流程

输入：

- 某张原图
- 该图目标 mask
- 场景高斯

输出：

- 高斯掩码概率

使用方式：

### 9.1 直接 3D 分割

- 阈值化后输出目标物体对应的高斯集合

### 9.2 约束 Semantic Gaussians 融合

在 `fusion.py` 中：

- 硬掩码：只给前景高斯写入 2D 特征
- 软掩码：按概率做加权融合

## 10. 与现有仓库的对应关系

### 10.1 可复用部分

- `sam3d_objects/semantic/model.py`
  - 可复用其 “2D + 3D + geometry” 融合思路
- `sam3d_objects/semantic/dataset.py`
  - 可复用数据组织骨架
- `sam3d_objects/semantic/loss.py`
  - 可复用 BCE/Dice/Lovasz 组件

### 10.2 需要新写的部分

- 高斯级 dataset
- SAM3D geometry cache loader
- 高斯属性编码器
- 高斯掩码预测模型
- 高斯级重投影损失

## 11. 评估指标

第一版推荐固定以下指标：

- `Gaussian IoU`
- `Gaussian F1`
- `Precision / Recall`
- `2D reprojection IoU`
- `background leakage ratio`
- `foreground hit ratio`

若用于 Semantic Gaussians 融合，还应补：

- 语义边界污染率
- 目标外区域特征泄漏率

## 12. 里程碑建议

### M1

仅用：

- 高斯属性
- 2D object token
- 强监督 `L_mask`

验证单视角条件高斯分割能否跑通。

### M2

加入：

- SAM3D geometry prior
- `L_geo`
- `L_reproj`

验证是否能减少背景污染与错投。

### M3

加入冻结 Sonata 特征。

验证复杂场景下是否进一步受益。

## 13. 一句话落地策略

第一版最重要的是：

**先把它实现成一个稳定的“单视角条件高斯分割器”，再让这个分割器去约束 `Semantic Gaussians` 的 feature fusion。**
