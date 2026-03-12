# Semantic Gaussians 几何约束投影调研

## 1. 问题定义

目标是改造 `semantic-gaussians` 的 2D 到 3D 语义投影过程，减少直接投影带来的错误归属：

- 前景物体特征被投到背景高斯上。
- 遮挡区域的像素特征被错误传播到不可见表面。
- 不同视角下的实例 ID 和边界不一致，导致多视角平均后特征被冲淡。
- 仅用深度阈值做可见性筛选，无法利用物体几何形状对投影位置做更强约束。

你的仓库已经具备两部分关键能力：

- `semantic-gaussians/fusion.py` 负责把 2D 特征投到高斯上。
- `sam3d_objects` 可以从单张图像和 mask 推理物体几何，并暴露 pointmap / voxel / shape latent 等几何先验。

因此最合理的改造路线不是重写整个 Semantic Gaussians，而是在投影阶段加入 `SAM3D geometry-aware gating`。

## 2. 当前仓库里的基线机制

### 2.1 Semantic Gaussians 当前的投影方式

当前 `semantic-gaussians/fusion.py` 的逻辑本质上是：

1. 用 2D 模型提取像素级特征。
2. 用 `PointCloudToImageMapper` 将每个高斯中心投到图像平面。
3. 用深度或渲染深度做可见性筛选。
4. 在投影像素位置直接读取特征。
5. 对多视角特征做简单累加和平均。

这个流程的优点是简单、训练代价低；缺点是它对下列信息建模不足：

- 物体边界
- 遮挡层级
- 实例一致性
- 物体自身几何形状

### 2.2 当前 mapper 的瓶颈

`semantic-gaussians/dataset/fusion_utils.py` 里的 `PointCloudToImageMapper` 只做：

- 投影坐标计算
- 图像内裁剪
- 基于深度差的 occlusion mask

它没有回答两个更关键的问题：

- 这个像素是否仍然位于目标物体的几何支撑区域内？
- 这个高斯是否应该接受来自该视角该像素的语义监督？

这正是可以由 `SAM3D` 几何补上的缺口。

## 3. 高相关论文与代码

下面只保留和你的目标直接相关的工作。

### 3.1 Semantic Gaussians

- 论文: Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting
- 时间: 2024
- 链接: <https://arxiv.org/abs/2403.15624>
- 代码: <https://github.com/sharinka0715/semantic-gaussians>
- 关系: 这是你当前要改造的基线。
- 启发: 它证明了 training-free / low-cost 的 2D feature lifting 是可行的，但当前 lifting 仍偏“投影后平均”。

### 3.2 Feature 3DGS

- 论文: Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields
- 时间: 2024
- 链接: <https://arxiv.org/abs/2312.03203>
- 代码: <https://github.com/ShijieZhou-UCLA/feature-3dgs>
- 关系: 它同样做 2D foundation feature 到 3DGS 的 lifting / distillation。
- 启发:
  - 说明高维语义特征绑定到高斯是有效方向。
  - 但它更偏 feature field distillation，不直接解决实例边界或 mask-level 错投问题。
  - 可借鉴其 feature-Gaussian 表达，但不能直接替代你想要的几何约束投影。

### 3.3 FiT3D

- 论文: Improving 2D Feature Representations by 3D-Aware Fine-Tuning
- 时间: 2024
- 链接: <https://arxiv.org/abs/2407.20229>
- 代码: <https://github.com/ywyue/FiT3D>
- 关系: 其核心是先把 2D feature lift 到 3D Gaussian 表示，再渲染回 2D 来做 3D-aware fine-tuning。
- 启发:
  - 它证明“先显式构建 3D-aware feature 再反向约束 2D 特征”是有效的。
  - 对你的任务更重要的不是 fine-tuning 本身，而是“3D geometry 能反过来纠正 2D 特征”的设计思路。

### 3.4 OpenMask3D

- 论文: OpenMask3D: Open-Vocabulary 3D Instance Segmentation
- 时间: 2023
- 链接: <https://arxiv.org/abs/2306.13631>
- 代码: <https://github.com/OpenMask3D/openmask3d>
- 关系: 它不是 3DGS 方法，但非常适合借鉴“实例级而不是像素级”的聚合思路。
- 启发:
  - 使用 3D instance mask proposal 作为聚合单元，而不是直接对所有点做平均。
  - 多视角图像特征先被聚合到候选实例，再做开放词汇匹配。
  - 如果你未来想让投影从 `per-gaussian average` 变成 `per-object constrained fusion`，这是高价值参考。

### 3.5 Panoptic Lifting

- 论文: Panoptic Lifting for 3D Scene Understanding with Neural Fields
- 时间: 2023
- 链接: <https://arxiv.org/abs/2212.09802>
- 项目页: <https://nihalsid.github.io/panoptic-lifting/>
- 关系: 不是 3DGS，但正面处理了“多视角 2D masks 不一致”这一核心问题。
- 启发:
  - 通过跨视角一致性和 assignment 机制解决 instance ID 漂移。
  - 适合借鉴到你的投影前后处理里，例如先建立 view-to-object 关联，再允许特征融合。

### 3.6 PCF-Lift

- 论文: PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion
- 时间: 2024
- 链接: <https://arxiv.org/abs/2410.10659>
- 代码: <https://github.com/Runsong123/PCF-Lift>
- 关系: 它针对的正是 noisy 2D segmentation 和 inconsistent IDs。
- 启发:
  - 用 probabilistic feature 表达不确定性，而不是把所有视角投票等价处理。
  - 适合迁移成 `per-view confidence weight` 或 `per-gaussian uncertainty`。

### 3.7 Unified-Lift

- 论文: Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting
- 时间: 2025
- 链接: <https://arxiv.org/abs/2503.14029>
- CVPR 2025 PDF: <https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Rethinking_End-to-End_2D_to_3D_Scene_Segmentation_in_Gaussian_Splatting_CVPR_2025_paper.pdf>
- 代码状态: 当前未检索到明确公开仓库
- 关系: 这是和你的目标最直接的论文之一。
- 启发:
  - 明确指出 direct matching 的 lifting 容易产生质量差的 3D 分割。
  - 引入 object-aware lifting 和 object-level codebook，而不是仅在高斯级别做点对点绑定。
  - 如果你后续不满足于“加一个几何过滤器”，而想做对象级建模，这篇是主参考。

### 3.8 FlashSplat

- 论文: FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally
- 时间: 2024
- 链接: <https://arxiv.org/abs/2409.08270>
- 代码: <https://github.com/florinshen/flashsplat>
- 关系: 它直接研究从 2D mask 到 3DGS segmentation 的映射。
- 启发:
  - 把 2D mask 到 3DGS 标签分配看成可求全局最优的问题，而不是依赖漫长梯度优化。
  - 对你的任务尤其有价值，因为你有明确的物体 mask 输入。
  - 可作为 `mask-constrained assignment` 的备选实现参考。

### 3.9 Gradient-Driven 3D Segmentation / Backprojection

- 论文: Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks
- 时间: 2024
- 链接: <https://arxiv.org/abs/2409.11681>
- 项目页: <https://jojijoseph.github.io/3dgs-segmentation/>
- 相关项目: Gradient-Weighted Feature Back-Projection, <https://jojijoseph.github.io/3dgs-backprojection/>
- 关系: 直接研究如何把 2D 掩码或特征回投到 3DGS。
- 启发:
  - 用 gradient-weighted / voting-based backprojection 而不是“像素命中即接受”。
  - 很适合改造成你的 feature projection weight 机制。

## 4. 对你最有价值的三条技术线

### 4.1 几何约束 gating

最贴合你当前设想的路线。

做法是：

- 用 SAM3D 生成 pointmap / shape latent / mesh proxy。
- 对每个高斯在某个视角上的投影，不只检查深度一致性，还检查它是否落在物体几何支持区域内。
- 只有通过几何一致性验证的投影才接收特征。

最接近的参考来源：

- FiT3D
- Feature 3DGS
- 你仓库里的 `sam3d_objects`

### 4.2 对象级 lifting

如果你发现“高斯级 gating”仍然无法解决跨视角错投，那么应提升到 object-aware lifting。

做法是：

- 先利用 mask 和几何把高斯聚成对象候选。
- 再对对象级别聚合多视角特征。
- 最后把对象特征回写给对象内的高斯。

最接近的参考来源：

- Unified-Lift
- OpenMask3D
- Panoptic Lifting
- PCF-Lift

### 4.3 全局优化或训练自由分配

如果你希望尽量不改动训练范式，可以考虑用 assignment / voting / optimal solver 替代简单平均。

做法是：

- 从每个视角导出候选高斯集合和匹配分数。
- 用全局优化或投票机制做高斯归属。
- 特征聚合基于归属结果，而不是直接像素采样平均。

最接近的参考来源：

- FlashSplat
- Gradient-Driven 3D Segmentation

## 5. 推荐的实现优先级

### P1: 最小侵入式改造

在当前 `fusion.py` 上加一个 `SAM3D geometry gate`：

- 输入: image, mask, SAM3D pointmap/shape
- 输出: per-gaussian per-view validity weight
- 优点: 改动小，能快速验证错误投影是否下降

### P2: 几何加权融合

将当前的简单平均改成带权重融合：

- 深度一致性权重
- SAM3D 几何一致性权重
- 距离 mask 边界的惩罚项
- 跨视角一致性置信度

### P3: 对象级投影

如果 P1/P2 效果有限，再做 object-aware lifting：

- 建立对象候选
- 对象级聚合特征
- 回写 per-gaussian semantic embedding

## 6. 结论

如果只问“有没有和你想法相近的现成方向”，答案是有，而且已经相当清晰：

- `Semantic Gaussians` 给出基线。
- `Unified-Lift` / `FlashSplat` / `Gradient-Driven 3DGS Segmentation` 直接针对 2D 到 3DGS 的 lifting 或分配问题。
- `OpenMask3D` / `Panoptic Lifting` / `PCF-Lift` 提供对象级、多视角一致性和噪声鲁棒性思路。
- `FiT3D` / `Feature 3DGS` 说明几何感知的 feature lifting 是合理路线。

对你当前仓库最现实的方案，不是整套复现这些方法，而是：

- 以 `Semantic Gaussians` 当前投影为底座；
- 用 `SAM3D` 的几何输出替换“仅靠深度判断是否接受特征”的规则；
- 再参考 `Unified-Lift` / `OpenMask3D` 决定是否进一步提升到对象级 lifting。
