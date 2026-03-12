# Semantic Gaussians + SAM3D 几何约束投影设计笔记

## 1. 面向当前仓库的插入点

当前最适合动手的位置有两个：

- `semantic-gaussians/dataset/fusion_utils.py`
- `semantic-gaussians/fusion.py`

推荐分工如下：

- `fusion_utils.py` 负责输出更强的 `mapping + weight + geometry_validity`
- `fusion.py` 负责把 `geometry_validity` 合并到最终特征聚合逻辑

### 1.1 更具体的代码落点

按当前代码结构，优先修改以下片段：

- `semantic-gaussians/fusion.py`
  - `65-70` 行: 初始化 `PointCloudToImageMapper`
  - `126-145` 行: 计算 mapping、取像素特征、写回高斯语义
- `semantic-gaussians/dataset/fusion_utils.py`
  - `30-78` 行: `compute_mapping()` 目前只返回 `(mapping, weight)`
- `sam3d_objects/semantic/model.py`
  - `337-410` 行: `extract_sam3d_features()`
  - `412-451` 行: `interpolate_shape_to_points()`

可操作建议是：

- 不直接把 `sam3d_objects/semantic/model.py` 搬进 `semantic-gaussians` 训练流程。
- 先抽出一个更轻量的 `SAM3DGeometryProjector`，只负责：
  - 输入 `image + mask`
  - 输出 `pointmap / shape / optional surface normals`

## 1.2 建议新增的数据结构

为避免在 `fusion.py` 中不断堆临时变量，建议把 mapper 输出从当前的：

- `mapping`
- `weight`

扩展为一个字典：

- `pixel_yx`
- `visible_mask`
- `depth_weight`
- `mask_weight`
- `surface_weight`
- `shape_weight`
- `fused_weight`

这样后续实验时可以很容易做 ablation。

## 2. 方案一: SAM3D Geometry Gate

这是最小侵入、最先应实现的版本。

### 2.1 输入

- RGB image
- 2D object mask
- SAM3D 输出:
  - pointmap
  - shape latent
  - 可选 mesh proxy
- Gaussian centers
- 当前视角相机参数

### 2.2 核心思想

当前规则是：

- 只要高斯投影到了一个像素，且深度差在阈值内，就接受该像素特征。

改成：

- 高斯投影到像素后，需要同时满足
  - 深度一致
  - 落在 mask 内或高置信物体区域内
  - 与 SAM3D pointmap / mesh 的局部表面距离足够小
  - 在 shape latent 中具有足够 occupancy score

### 2.3 实现形式

给每个高斯-视角对定义一个权重：

`w = w_depth * w_mask * w_surface * w_shape`

其中：

- `w_depth`: 当前已有的深度一致性
- `w_mask`: 是否落在物体 mask 内，或者到 mask 边界的距离衰减
- `w_surface`: 高斯中心到 SAM3D pointmap 反投影表面的距离权重
- `w_shape`: shape latent 采样得到的 occupancy / inside-object 分数

最终不用 `+= features`，而是：

- `semantic_sum += w * feature`
- `weight_sum += w`
- 最后 `semantic = semantic_sum / max(weight_sum, eps)`

### 2.3.1 在当前代码中的替换方式

直接对应当前 `fusion.py` 的替换关系：

- 旧逻辑:
  - `mapping = ...`
  - `mask = mapping[:, 3]`
  - `features_mapping = features[:, mapping[:, 1], mapping[:, 2]]`
  - `gaussians._times[mask_k] += 1`
  - `gaussians._features_semantic[mask_k] += features_mapping[mask_k]`

- 新逻辑:
  - `mapping_dict = mapper.compute_mapping_with_geometry(...)`
  - `weights = mapping_dict["fused_weight"]`
  - `valid = weights > 0`
  - `gaussians._times[valid] += weights[valid]`
  - `gaussians._features_semantic[valid] += weights[valid, None] * features_mapping[valid]`

其中 `gaussians._times` 不再表示“命中次数”，而表示“累积总权重”。

### 2.4 预期收益

- 减少前景特征泄漏到背景
- 边界附近错投更少
- 遮挡区域更稳

### 2.5 失败模式

- SAM3D 几何本身出错时，可能把可用监督过滤掉
- 单张图像推理的 pointmap 较粗时，过强 gating 会导致监督太稀

因此建议先把它做成 soft weight，而不是 hard reject。

## 3. 方案二: Geometry-Aware Weighted Fusion

这是在方案一基础上的增强版。

### 3.1 核心思想

不是只判断“能不能投”，而是判断“该不该信”。

每个视角的特征贡献不仅由几何一致性决定，还由跨视角一致性决定。

### 3.2 新增权重来源

- `w_view_consistency`
  - 同一个高斯在多个视角上得到的特征应当相似
  - 如果某个视角给出的特征与已有均值偏差过大，则降低它的权重
- `w_boundary_penalty`
  - 距离 mask 边界越近，权重越低
- `w_angle`
  - 视线方向与 SAM3D 局部法线夹角过大时，降低权重

### 3.3 最适合借鉴的文献

- PCF-Lift
- Gradient-Weighted Feature Back-Projection
- Panoptic Lifting

### 3.4 何时做

如果方案一已经显著降低错误投影，就先停在这里。

如果方案一仍有明显跨视角漂移，再引入这一步。

### 3.5 一个简单可落地的 consistency 版本

先不要上复杂的 probabilistic fusion，先做一个在线均值版本：

1. 某个高斯第一次接收到特征时，直接写入。
2. 后续视角到来时，计算当前视角特征与已聚合特征的 cosine similarity。
3. 如果相似度低于阈值，则把该视角的 `w_view_consistency` 降低。

这样可以在几乎不改训练框架的前提下，抑制明显离群的错误投影。

## 4. 方案三: Object-Aware Lifting

这是更重的结构性改造。

### 4.1 核心思想

从 “per-gaussian feature averaging” 升级到 “per-object constrained aggregation”。

先把高斯聚到对象，再对对象聚合语义，再回写高斯。

### 4.2 一个适合当前仓库的版本

1. 用 SAM3D 的几何输出和 2D mask 建立 object support region。
2. 找到与该对象 region 高重合的高斯集合。
3. 所有视角特征先聚合到对象原型 embedding。
4. 再把对象 embedding 回写到对象内高斯，或者与 per-gaussian embedding 融合。

### 4.3 参考来源

- Unified-Lift
- OpenMask3D
- FlashSplat

### 4.4 优点

- 对视角不一致更稳
- 对实例边界更稳
- 更适合后续开放词汇检索和实例查询

### 4.5 代价

- 需要显式对象建模
- 数据结构和训练流程都会更复杂

## 5. 建议的实施顺序

### 阶段 1

实现 `SAM3D geometry gate`：

- 不改训练方式
- 不引入对象级代码本
- 只替换当前简单平均投影

具体输出物建议是：

- 一个新 mapper 接口
- 一个最小 YAML 配置开关，例如
  - `fusion.use_sam3d_geometry`
  - `fusion.geometry_weight_mode`
  - `fusion.mask_erode_pixels`
  - `fusion.shape_weight_threshold`

### 阶段 2

加入 `geometry-aware weighted fusion`：

- 增加边界权重
- 增加 view consistency 权重
- 观察多视角稳定性

这一阶段最适合做消融：

- depth only
- depth + mask
- depth + mask + pointmap
- depth + mask + pointmap + shape

### 阶段 3

如果还不够，再做 `object-aware lifting`：

- 增加对象原型
- 对象级聚合特征
- 必要时引入 assignment 或 optimal solver

## 6. 建议验证指标

### 定量

- 2D 重渲染语义一致性
- 目标 mask 内平均特征相似度
- mask 外泄漏率
- 跨视角同一高斯特征方差

### 针对当前任务最实用的附加指标

- `foreground hit ratio`
  - 接收特征的高斯中，真正落在目标物体支持区域内的比例
- `background leakage ratio`
  - 目标 mask 提供的特征被写到背景高斯的比例
- `boundary contamination`
  - 距离 mask 边界一定像素范围内的高斯污染程度

### 定性

- 物体边界是否更干净
- 细长结构和遮挡处是否更稳
- 背景污染是否减少

## 7. 最终建议

如果你的目标是尽快做出一个能工作的改进版本，应优先实现：

- `SAM3D pointmap/shape latent -> per-gaussian geometry weight`
- `weighted feature fusion instead of raw averaging`

如果这个版本有效，再考虑：

- `object-aware lifting`
- `optimal assignment`

这条路线最符合当前仓库现状，也最接近你最初提出的“给定图像和掩码，通过 SAM3D 推理几何形状，再用几何形状约束投影位置”的想法。

## 8. 建议保留的假设

为避免第一版过重，建议先固定这些假设：

- 只处理单个目标 mask
- 一次只投影一个物体
- SAM3D 几何仅作为 soft constraint
- 不在第一版引入训练式 object codebook
- 不修改 `distill.py`，先只改投影前处理
