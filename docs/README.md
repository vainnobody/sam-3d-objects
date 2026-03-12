# Docs Index

本目录收集了基于 `Semantic Gaussians + SAM3D geometry-aware projection` 的调研结果与实现草案。

## 文件说明

- `semantic_gaussian_projection_survey.md`
  - 主综述
  - 解释当前 `Semantic Gaussians` 投影机制的瓶颈
  - 总结和任务最相关的论文与代码

- `semantic_gaussian_projection_catalog.md`
  - 条目化清单
  - 快速查看每篇工作是否开源、依赖什么信号、和当前任务的相关性

- `semantic_gaussian_projection_design_notes.md`
  - 面向实现的设计笔记
  - 重点描述如何把 `SAM3D` 的几何输出接入 `semantic-gaussians`

- `geoalign_lift_idea.md`
  - 新的方法提案
  - 将任务具体化为 `单视角条件高斯分割`
  - 详细描述轻量级模型结构、输入高斯特征、损失设计与如何接回 `Semantic Gaussians`

- `geoalign_lift_lite_spec.md`
  - 实现规格
  - 将方法拆成数据、几何先验、模型、损失、训练流程
  - 可直接作为第一版原型开发清单

## 当前仓库里的关键代码位置

- 投影与特征融合:
  - `semantic-gaussians/fusion.py`
- 3D-2D 投影与可见性:
  - `semantic-gaussians/dataset/fusion_utils.py`
- SAM3D 几何与 shape latent 提取:
  - `sam3d_objects/semantic/model.py`

## 推荐阅读顺序

1. 先读 `semantic_gaussian_projection_survey.md`
2. 再看 `semantic_gaussian_projection_catalog.md`
3. 再读 `geoalign_lift_idea.md` 确定整体方法
4. 读 `geoalign_lift_lite_spec.md` 确认实现边界
5. 最后用 `semantic_gaussian_projection_design_notes.md` 对照代码实现
