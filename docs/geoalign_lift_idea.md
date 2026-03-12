# GeoAlign-Lift-Lite: 轻量级单视角条件高斯分割模型

## 1. 任务重新定义

目标不再表述为“学习一个 soft projection weight 场”，而是更直接地定义为：

> 输入一张用于重建 3DGS 的原始图像、该图上的目标 2D mask，以及对应场景的 3DGS 高斯集合，输出该目标物体在 3DGS 场景中的高斯掩码。

这个输出既可以解释为：

- 该物体的 `3D 分割结果`
- 该物体的 `高斯掩码`
- 或者后续语义投影时的 `有效接收高斯集合`

因此这个方法本质上是一个：

`view-conditioned object segmentation on 3D Gaussians`

而不是通用 3D 语义分割。

## 2. 为什么这样定义更合适

相较于直接优化 `Semantic Gaussians` 的特征投影流程，先训练一个“条件高斯分割模型”有三个优点：

### 2.1 监督目标更清晰

我们真正想知道的是：

- 当前图像中的这个物体，对应场景中的哪些高斯？

这天然就是一个二值分割问题，而不是抽象的投影权重回归问题。

### 2.2 更容易轻量化

如果目标是直接输出高斯掩码，那么模型只需要：

- 对 2D 目标做条件编码
- 对 3DGS 高斯做轻量编码
- 用几何先验约束两者对齐

不需要第一版就做复杂的多视角联合优化。

### 2.3 更方便接回 Semantic Gaussians

一旦得到了高斯掩码，后续在 `fusion.py` 中的用法非常直接：

- 只有掩码内高斯接收该视角的 2D 特征
- 或者掩码概率作为 soft weight 做加权融合

所以高斯分割是一个更稳的中间表征。

## 3. 方法概述

我建议的方法名称为：

`GeoAlign-Lift-Lite`

它是一个轻量级的单视角条件高斯分割模型，整体流程如下：

1. 从输入图像和 mask 中提取目标感知的 2D 表示。
2. 从场景高斯中提取每个高斯的轻量级结构特征。
3. 利用 SAM3D 从同一 `(image, mask)` 中推理目标几何。
4. 将几何先验附加到每个高斯上。
5. 通过轻量级 cross-attention 或 gated MLP，预测每个高斯是否属于当前目标。

输出：

- `gaussian_mask_logits`
- `gaussian_mask_probs`

## 4. 输入与输出

### 4.1 输入

每个训练样本由以下内容组成：

- `I`: 一张重建 3DGS 所使用的原始 RGB 图像
- `M`: 该图中目标物体的 2D mask，由 SAM 或其他 2D 分割器提供
- `G = {g_i}`: 当前场景的 3DGS 高斯集合
- `C`: 当前图像对应的相机参数
- `Y_3d`: 可选的高斯级真值掩码

其中每个高斯 `g_i` 至少包含：

- `xyz`
- `color / features_dc`
- `opacity`
- `scale`
- `rotation`

可选增加：

- `已有 semantic slot`
- `从高斯邻域提取的局部几何统计`

### 4.2 输出

模型主输出固定为：

- `p_i in [0,1]`: 第 `i` 个高斯属于当前目标的概率

辅助输出可选为：

- `logit_i`
- `rendered_mask_2d`
- `geometry_score_i`

最终二值高斯掩码由阈值得到：

- `m_i = 1[p_i > tau]`

## 5. 模型结构

模型采用三分支结构，但所有重 backbone 默认冻结，只训练轻量头。

### 5.1 2D 目标编码分支

输入：

- `I`
- `M`

作用：

- 提取当前 mask 所对应的目标外观表示

推荐实现：

- 使用冻结的 DINOv2 或 CLIP vision encoder
- 先提取 patch tokens
- 再用 mask pooling 得到 `object token`

第一版不建议把 mask 作为第 4 通道重新训练 2D backbone，因为这会加重模型。

输出：

- `t_obj in R^C`

### 5.2 3D 高斯编码分支

输入：

- 场景中所有高斯的属性

作用：

- 把高斯表示映射到适合做条件分割的紧凑特征空间

推荐的轻量特征设计如下：

#### 必选特征

- `xyz`
- `rgb` 或 `features_dc`
- `opacity`
- `log_scale`
- `rotation` 的紧凑参数化

#### 可选特征

- 与相机的相对深度
- 投影坐标
- 高斯局部密度

#### Sonata 特征的使用方式

你希望使用点云特征，但又要轻量。

因此推荐：

- `不把 Sonata 当作可训练主干`
- 把 Sonata 当作冻结的 3D 特征提取器
- 对场景中与高斯中心对应的点云或采样点提取 `512-dim` 特征
- 再经过一个小投影层降到 `64` 或 `128` 维

如果工程成本太高，第一版可以先不接 Sonata，只使用高斯属性和 SAM3D 几何特征。

输出：

- `h_i^g in R^D`

### 5.3 SAM3D 几何先验分支

输入：

- 同一个 `I, M`

作用：

- 提供该目标物体在 3D 中的粗几何支持区域

推荐从 SAM3D 提取：

- `shape latent`
- `pointmap`
- `voxel occupancy`

再将这些几何结果映射到高斯级别：

- 若高斯位于 SAM3D 估计的目标几何内部，则几何支持高
- 若高斯远离 pointmap 对应表面，则几何支持低

具体可构造的高斯几何特征包括：

- `shape occupancy score`
- `distance to pointmap surface`
- `inside/outside voxel flag`
- `surface confidence`

输出：

- `h_i^geo in R^G`

## 6. 融合头设计

为了保持轻量级，融合头只训练一个小模块。

### 6.1 推荐版本

对每个高斯构造：

`z_i = MLP([h_i^g ; h_i^geo ; PE(xyz_i)])`

再用 object token 作为条件信息：

`a_i = CrossAttention(t_obj, z_i)` 或 `a_i = z_i * sigma(W t_obj)`

最后：

`logit_i = MLP([z_i ; a_i])`

`p_i = sigmoid(logit_i)`

### 6.2 极简版

如果要进一步轻量化，可以完全不用多层 transformer，只用：

- 一个 object token
- 一个 gated MLP

例如：

`gate = sigmoid(W_g t_obj)`

`z_i' = z_i * gate`

`p_i = sigmoid(MLP(z_i'))`

这比 full cross-attention 更轻，适合第一版原型。

## 7. 训练监督

训练采用“强监督主导，几何和重投影辅助”的策略。

### 7.1 主监督: 高斯级二值分割损失

如果样本具备高斯级真值掩码：

- 使用 `BCE + Dice`
- 或 `BCE + Lovasz`

主损失为：

- `L_mask`

这是第一优先级监督。

### 7.2 几何先验一致性损失

即便没有完美 3D 真值，也可以利用 SAM3D 约束预测：

- 高概率高斯应与目标 shape occupancy 一致
- 高概率高斯不应远离 pointmap 所支持的物体表面

具体可分为：

- `L_occ`
- `L_surface`

### 7.3 2D 重投影损失

将预测的高斯掩码投影回当前视角，得到一个 soft 2D mask：

- 该 mask 应与输入的 2D 目标 mask 对齐

损失：

- `L_reproj`

这个损失很重要，因为它保证模型虽然在 3D 空间输出掩码，但仍然忠于当前这张图像的目标条件。

### 7.4 稀疏性正则

由于单视角条件分割容易过扩张，建议加入：

- `L_sparse`

作用：

- 避免模型把大片高斯都预测为前景

### 7.5 总损失

第一版推荐：

`L = L_mask + lambda_geo * (L_occ + L_surface) + lambda_reproj * L_reproj + lambda_sparse * L_sparse`

若没有高斯级真值，则第一版可以退化为：

`L = lambda_geo * (L_occ + L_surface) + lambda_reproj * L_reproj + lambda_sparse * L_sparse`

但训练稳定性会弱很多，因此更推荐至少有一部分强监督样本。

## 8. 训练样本组织方式

根据你的偏好，第一版固定为：

- `单视角条件分割`

即每个样本只包含：

- 一张图
- 一个目标 mask
- 整个场景的高斯集合

而不是多视角联合输入。

这样有几个好处：

- 更轻量
- 更好训练
- 更容易复用现有数据组织
- 更适合先做 proof-of-concept

多视角一致性可以留到第二阶段：

- 训练时作为辅助损失
- 推理时对多个视角预测结果做融合

## 9. 数据构造建议

### 9.1 基础数据

每个场景需要：

- `3DGS 高斯`
- 原始训练图像
- 每张图对应的相机参数
- 每张图上若干目标 mask

### 9.2 mask 来源

mask 可以直接来自：

- SAM automatic mask
- 或人工点击/框选后用 SAM 得到的目标 mask

第一版默认采用 SAM 自动或半自动生成。

### 9.3 高斯真值来源

如果没有直接的高斯级标注，可以构造伪标签：

1. 将当前图像的 2D mask 投影到 3DGS 场景中，得到候选高斯。
2. 用深度一致性筛掉明显错误匹配。
3. 再用 SAM3D 几何筛掉不在目标几何支持区域内的候选。
4. 将剩余候选作为高精度伪正样本，外部区域作为伪负样本。

这样模型就不是从零学，而是学习“如何修正粗投影伪标签”。

## 10. 点云特征设计建议

你特别提到“点云特征需要设计”，并且希望轻量。

这里给出一个分阶段设计。

### 10.1 第一版推荐

使用以下高斯级输入：

- `xyz`
- `rgb / dc color`
- `opacity`
- `log scale`
- `rotation`
- `SAM3D geometry score`

不接 Sonata。

优点：

- 实现简单
- 显存小
- 能快速验证几何先验是否有效

### 10.2 第二版增强

加入冻结的 Sonata 特征：

- 从高斯中心附近采样点云
- 用 Sonata 提取局部场景特征
- 用小线性层压到低维
- 与高斯属性拼接

优点：

- 场景上下文更强
- 对遮挡和相似外观的歧义更稳

### 10.3 不推荐的第一版设计

不建议第一版：

- 端到端训练 Sonata
- 加多层 transformer
- 同时做多视角联合建模

因为这会让“轻量级”目标失效。

## 11. 和 Semantic Gaussians 的集成方式

训练好 `GeoAlign-Lift-Lite` 后，接回 `Semantic Gaussians` 的方式很直接。

### 11.1 硬掩码版本

- 对于当前视角，只把 2D 特征写到预测前景高斯上

即：

- `if m_i == 1: gaussian_i receives feature`

### 11.2 软权重版本

- 用 `p_i` 作为 feature fusion 权重

即：

`feature_sum_i += p_i * feature_2d(pixel_i)`

`weight_sum_i += p_i`

这个版本更稳，也更容易保留边界过渡信息。

### 11.3 最终收益

这一步的目的不是替换整个 `Semantic Gaussians`，而是把原本粗糙的直接投影换成：

- `2D 条件下的 3D 高斯选择`

从而显著减少：

- 背景泄漏
- 遮挡错投
- 边界污染

## 12. 与已有工作之间的关系

### 与 Semantic Gaussians

- 它是直接扩展其 `fusion` 阶段的前置分割模块。

### 与 OpenMask3D / Unified-Lift

- 相同点：都在做从 2D 目标到 3D 对象的条件对齐。
- 不同点：这里的目标载体是 `3DGS 高斯`，且强调轻量单视角条件分割。

### 与 FlashSplat

- FlashSplat 更偏 2D mask 到 3DGS 的优化式全局分配。
- 这里是可训练版本，重点是学习从粗投影到正确高斯掩码的映射。

### 与 FiT3D / Feature 3DGS

- 它们说明 2D 表示和 3D Gaussian 表示之间的桥接是合理的。
- 本方法进一步把桥接具体化成“条件高斯分割”。

## 13. 第一版可行配置

为了保证真正轻量，推荐第一版配置固定如下：

- 单视角输入
- 直接分割高斯
- 冻结 DINOv2
- 冻结 SAM3D
- 高斯属性作为主 3D 输入
- SAM3D shape / pointmap 作为几何先验
- 融合头只训练 1-2 层 MLP 或单层 cross-attention
- 若工程复杂度过高，Sonata 延后到第二版

## 14. 这篇方法的核心一句话

> GeoAlign-Lift-Lite does not directly project 2D object features onto 3D Gaussians. Instead, it first predicts a view-conditioned Gaussian mask using image appearance, Gaussian attributes, and SAM3D geometric priors, and then uses this mask to control semantic lifting in Semantic Gaussians.

## 15. 总结

这个 idea 的本质是：

- 把“错误投影”问题转化为“给定 2D 目标，预测该目标在 3DGS 中的高斯掩码”
- 用 SAM3D 提供几何先验
- 用轻量级条件分割头做高斯级对齐
- 再用这个高斯掩码去约束 `Semantic Gaussians` 的语义融合

如果目标是训练一个足够轻、但确实能改善投影质量的模型，这条路线比直接做大规模多视角联合模型更合理，也更适合作为第一版。
