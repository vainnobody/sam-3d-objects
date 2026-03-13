# 在 LERF-OVS 上验证 Semantic Gaussians（SG）

本文档面向当前仓库中的 `semantic-gaussians/` 子项目，目标是：

1. 在 `LERF-OVS` 场景上训练/准备 RGB 3D Gaussian；
2. 运行 `fusion.py` 得到 SG 的 open-vocabulary 高斯特征；
3. 使用本仓库新增的 `eval_lerf_ovs.py` 按 **LangSplat / LERF-OVS benchmark 口径**计算：
   - `mean IoU`
   - `localization accuracy`

> 这份文档解决的是“**如何在 LERF-OVS 上定量验证 semantic-gaussians**”，不是只做 viewer 定性展示。

---

## 1. 先理解为什么不能直接用现有 eval

当前仓库里已有两个看起来相关但**不能直接拿来做 LERF-OVS benchmark** 的入口：

### 1.1 `semantic-gaussians/eval_segmentation.py`

这个脚本是 **ScanNet 风格语义分割评测**，它依赖：

- 固定数据集标签集（`scannet20` / `cocomap`）
- `label-filt` / `color` 这样的 ScanNet 目录组织
- 类别映射表 `dataset/scannet/scannetv2-labels.modified.tsv`

LERF-OVS 不是这套格式。它是：

- COLMAP 场景 + `images/` + `sparse/0/*.bin`
- 单独的 `label/<scene>/frame_*.json` 标注
- 每张图是一组 **文本 prompt/object category + polygon mask + bbox**

所以 **不能直接跑 `eval_segmentation.py`**。

### 1.2 LangSplat 官方 `eval/eval.sh`

LangSplat 的官方 benchmark 脚本依赖它自己的：

- 3-level render features
- autoencoder
- feature decoder

而 `semantic-gaussians/fusion.py` 输出的是：

- `0.pt`
- 其中包含 `feat` 与 `mask_full`

这和 LangSplat 的中间表征不同，因此 **也不能直接拿 `eval.sh` 去吃 SG 的 `0.pt`**。

### 1.3 正确做法

正确方案是：

```text
LERF-OVS scene
  -> semantic-gaussians/train.py
  -> semantic-gaussians/fusion.py
  -> semantic-gaussians/eval_lerf_ovs.py
```

也就是：

- 用 SG 生成高斯语义特征；
- 用新的 `eval_lerf_ovs.py` 直接把这些高斯特征渲染成 prompt relevancy map；
- 按 LangSplat / LERF-OVS 的指标定义重新计算 IoU 和 localization accuracy。

---

## 2. 数据要求

### 2.1 场景数据

你已经可以用：

```bash
python scripts/download_lerf_ovs.py --scene teatime
python scripts/download_lerf_ovs.py --scene all
```

默认解压到：

```text
data/lerf_ovs/<scene>
```

例如：

```text
data/lerf_ovs/teatime/
  images/
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin
```

### 2.2 benchmark GT 标注

做 LERF-OVS benchmark **还必须额外准备 label 标注目录**：

```text
data/lerf_ovs/label/<scene>/
  frame_00001.json
  frame_00001.jpg
  frame_00002.json
  frame_00002.jpg
  ...
```

这是 LangSplat 官方 benchmark 也要求的输入。

如果没有 `data/lerf_ovs/label`，新的 `eval_lerf_ovs.py` 会直接报错。

### 2.3 推荐最终目录结构

```text
sam-3d-objects/
  data/
    lerf_ovs/
      figurines/
        images/
        sparse/0/
      ramen/
        images/
        sparse/0/
      teatime/
        images/
        sparse/0/
      waldo_kitchen/
        images/
        sparse/0/
      label/
        figurines/
          frame_*.json
          frame_*.jpg
        ramen/
          frame_*.json
          frame_*.jpg
        teatime/
          frame_*.json
          frame_*.jpg
        waldo_kitchen/
          frame_*.json
          frame_*.jpg
```

---

## 3. 环境要求

建议在 Linux + NVIDIA GPU 上执行，并使用 `semantic-gaussians/` 自己的环境。

```bash
cd semantic-gaussians
conda env create -f environment.yml
conda activate sega
pip install -r requirements.txt
```

如果你之前已经跑通过 `train.py` / `fusion.py`，一般不需要额外补环境。

### 3.1 关于 2D 模型

本 runbook 的默认选择是：

- `OpenSeg` for fusion/eval

原因：

- 当前 `semantic-gaussians` 原始配置更偏向 `openseg`
- 新 evaluator 的默认配置也与之对齐

如果你的环境里 TensorFlow / OpenSeg 不稳定，可以改成 `lseg`，但要保证：

- fusion 时用 `lseg`
- eval 时也用 `lseg`

**两边必须一致。**

---

## 4. 第一步：训练 RGB Gaussian

下面给出推荐目录组织。为了不和原始 `output/` 混在一起，建议把 LERF-OVS 的训练结果放在：

```text
semantic-gaussians/output/lerf_ovs/<scene>/
```

### 4.1 单场景训练示例：`teatime`

在 `semantic-gaussians/` 目录下执行：

```bash
python train.py \
  scene.scene_path=../data/lerf_ovs/teatime \
  scene.colmap_images=images \
  scene.test_cameras=False \
  train.exp_name=lerf_ovs/teatime
```

更稳妥的推荐方式是直接让 `train.py` 使用独立输出名，并在文档里把产物整理为：

```text
semantic-gaussians/output/lerf_ovs/teatime/
  point_cloud/
    iteration_30000/
      point_cloud.ply
```

如果你只想做 smoke test，可把 `train.iterations` 先降到 `7000`；
如果你要做 benchmark，建议按较完整训练设置跑完。

### 4.2 训练通过的标志

至少应出现：

```text
output/lerf_ovs/teatime/point_cloud/iteration_xxxxx/point_cloud.ply
```

没有这个 `.ply`，后续 fusion/eval 都不能继续。

---

## 5. 第二步：运行 SG fusion

目标是在训练好的高斯上投影 2D 语义特征，输出：

```text
semantic-gaussians/fusion_lerf_ovs/<scene>/0.pt
```

### 5.1 推荐命令

```bash
python fusion.py \
  scene.scene_path=../data/lerf_ovs/teatime \
  scene.colmap_images=images \
  scene.test_cameras=False \
  model.model_dir=./output/lerf_ovs/teatime \
  fusion.out_dir=./fusion_lerf_ovs/teatime \
  fusion.model_2d=openseg \
  fusion.img_dim=[<width>,<height>]
```

> 注意：当前 `fusion.py` 默认从 `config/fusion_scannet.yaml` 读取基础配置，因此你需要像上面那样**显式覆盖** `scene/model/fusion` 相关字段，尤其是 `fusion.img_dim`。

### 5.2 输出检查

成功后至少会得到：

```text
fusion_lerf_ovs/teatime/0.pt
```

这个文件中包含：

- `feat`: 被赋值的高斯语义特征
- `mask_full`: 哪些高斯有 feature

---

## OpenSeg / TensorFlow / cuDNN 常见环境故障

如果你在 `MODEL_2D=openseg` 的 fusion / eval 过程中遇到 TensorFlow 或 cuDNN 相关报错，优先看这一节。

### 典型报错 1：OpenSeg 在 `serving_default` 处失败

常见现象：

```text
tensorflow.python.framework.errors_impl.UnimplementedError: Graph execution error
Detected at node efficientnet-b7/.../Conv2D
DNN library is not found
```

这类错误通常发生在：

- `semantic-gaussians/fusion.py`
- `semantic-gaussians/model/openseg_predictor.py`

具体是 OpenSeg 的 TensorFlow SavedModel 在执行：

```python
self.model.signatures["serving_default"](...)
```

时，进入 EfficientNet backbone 的卷积算子，但当前环境里的 **cuDNN / CUDA 运行时不可用或不匹配**，所以 TensorFlow 图执行失败。

### 典型报错 2：重装 TensorFlow 后，`import torch` 失败

常见现象：

```text
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

并且报错发生在：

```python
from torch._C import *
```

这通常说明你重装 TensorFlow 或 CUDA 相关依赖后，把同一个环境里的 **PyTorch / TensorFlow / CUDA / cuDNN** 版本关系打乱了。  
PyTorch 现在尝试加载 `libcudnn.so.9`，但当前环境里没有这个版本，或者动态库搜索路径里找不到它。

### 当前仓库预期的环境基线

`semantic-gaussians/` 目录下的推荐基线是：

- `semantic-gaussians/environment.yml`
  - `python=3.9.18`
  - `pytorch=2.1.1`
  - `pytorch-cuda=11.8`
- `semantic-gaussians/requirements.txt`
  - `tensorflow[and-cuda]==2.14.0`

如果你直接在现有环境里多次 `pip install` / `pip uninstall` TensorFlow、torch、nvidia-* 相关包，极容易出现：

- TensorFlow 能 import，但 OpenSeg 推理时报 `DNN library is not found`
- TensorFlow 装好后，PyTorch 反而因为 `libcudnn.so.X` 找不到而无法 import

### 推荐排查顺序

先运行仓库里新增的诊断脚本：

```bash
bash semantic-gaussians/tools/diagnose_openseg_env.sh
```

如果你想额外验证 OpenSeg SavedModel 的最小推理链路：

```bash
bash semantic-gaussians/tools/diagnose_openseg_env.sh \
  --run-serving-test \
  --image /path/to/test.jpg
```

然后按顺序检查：

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 推荐修复策略

#### 正式方案：重建干净环境

最稳妥的方式不是继续在当前环境里补 `libcudnn.so.9`，而是**重建一个干净的 `semantic-gaussians` conda 环境**，然后按仓库预期重新安装：

```bash
cd semantic-gaussians
conda env create -f environment.yml
conda activate sega
pip install -r requirements.txt
```

关键原则：

- 不要在这个环境里额外安装另一版 `torch` / `torchvision`
- 不要手工混装多套 `cudnn` / `nvidia-*` pip 包
- 尽量让 `PyTorch + pytorch-cuda + TensorFlow` 保持和仓库文档一致

#### 临时绕过方案：改用 LSeg

如果你当前只是想先把 benchmark 主流程跑通，而不是立刻修好 OpenSeg，可以直接切到 `lseg`：

```bash
MODEL_2D=lseg bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
```

如果 3DGS 已经训练好，只想重跑 fusion + eval：

```bash
MODEL_2D=lseg RUN_TRAIN=0 bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
```

### 经验结论

- `DNN library is not found` 基本不是 `train.py / fusion.py / eval_lerf_ovs.py` 的逻辑 bug，而是 **TensorFlow OpenSeg GPU 运行时问题**
- `libcudnn.so.9` 缺失基本不是单个 Python 文件的问题，而是 **环境依赖栈冲突**
- 如果你希望长期保留 `MODEL_2D=openseg`，优先选择 **重建一致环境**
- 如果你只想先完成实验，`MODEL_2D=lseg` 是最省时的 fallback

---

## 6. 第三步：运行 LERF-OVS benchmark evaluator

本仓库新增脚本：

```text
semantic-gaussians/eval_lerf_ovs.py
```

它直接读取：

- 训练好的高斯模型
- `fusion.py` 输出的 `0.pt`
- `data/lerf_ovs/label/<scene>/frame_*.json`

然后输出：

- `mean IoU`
- `localization accuracy`
- 每个 prompt 的 heatmap / composited / chosen mask / localization overlay

### 6.1 默认配置文件

新增配置：

```text
semantic-gaussians/config/eval_lerf_ovs.yaml
```

默认内容假设：

- 场景根目录：`../data/lerf_ovs`
- 模型根目录：`./output/lerf_ovs`
- fusion 根目录：`./fusion_lerf_ovs`
- GT 标注根目录：`../data/lerf_ovs/label`

### 6.2 单场景评测示例

```bash
python eval_lerf_ovs.py \
  eval.scene_names='[teatime]' \
  scene.scene_path=../data/lerf_ovs \
  model.model_dir=./output/lerf_ovs \
  fusion.out_dir=./fusion_lerf_ovs \
  eval.label_root=../data/lerf_ovs/label \
  eval.output_dir=./eval_result_lerf_ovs \
  eval.model_2d=openseg \
  eval.mask_thresh=0.4
```

### 6.3 四个场景一起评

```bash
python eval_lerf_ovs.py \
  eval.scene_names='[figurines,ramen,teatime,waldo_kitchen]' \
  scene.scene_path=../data/lerf_ovs \
  model.model_dir=./output/lerf_ovs \
  fusion.out_dir=./fusion_lerf_ovs \
  eval.label_root=../data/lerf_ovs/label \
  eval.output_dir=./eval_result_lerf_ovs \
  eval.model_2d=openseg \
  eval.mask_thresh=0.4
```

---

## 7. evaluator 具体做了什么

新增 evaluator 的行为是固定的：

1. 读取 scene 的 3DGS：
   - `output/lerf_ovs/<scene>/point_cloud/iteration_x/point_cloud.ply`
2. 读取 fusion feature：
   - `fusion_lerf_ovs/<scene>/0.pt`
3. 把高斯 feature 填回 `gaussians._features_semantic`
4. 读取 GT json 中的所有 prompt/object
5. 用与 fusion 一致的 text model 提取 prompt 文本 embedding
6. 将高斯语义特征渲染回 GT 分辨率
7. 对每个 prompt 计算 similarity / relevancy map
8. 基于平滑后的 relevancy map 计算：
   - mask threshold 后的 IoU
   - 峰值点是否落在 GT bbox 中

### 7.1 指标定义

#### IoU

- 先对 prompt relevancy map 做 min-max normalization 到 `[0,1]`
- 再做均值滤波平滑
- 然后按 `mask_thresh=0.4` 二值化
- 与 GT polygon mask 计算 IoU

#### Localization accuracy

- 取平滑后 relevancy map 的峰值点
- 如果这个点落在该 prompt 的任意一个 GT bbox 内，则视为命中
- 场景级 accuracy = 命中 prompt 数 / 总 prompt 数

这与 LangSplat 官方 LERF-OVS eval 的思路一致，但输入特征换成了 SG 的高斯 feature field。

---

## 8. 输出结果怎么看

评测完成后，结果目录形如：

```text
semantic-gaussians/eval_result_lerf_ovs/
  summary.json
  summary.txt
  teatime/
    metrics.json
    metrics.txt
    frame_00001/
      chosen_<prompt>.png
      gt/<prompt>.png
      heatmap/<prompt>.png
      composited/<prompt>.png
      localization/<prompt>.png
```

### 8.1 `metrics.json`

scene 级主要字段：

- `mean_iou`
- `localization_accuracy`
- `num_frames`
- `num_prompts`

### 8.2 `summary.json`

包含所有 scene 的聚合结果，以及 overall 汇总。

### 8.3 可视化文件用途

- `gt/<prompt>.png`: GT mask
- `chosen_<prompt>.png`: evaluator 预测 mask
- `heatmap/<prompt>.png`: prompt relevancy heatmap
- `composited/<prompt>.png`: heatmap 与原图融合
- `localization/<prompt>.png`: bbox + 峰值点可视化

这些文件用于快速判断：

- 是 feature 没投影好；
- 还是 prompt 文本匹配偏差；
- 还是 threshold / smoothing 不合适。

---

## 9. 推荐执行顺序

建议先从单场景 `teatime` 打通，再扩展到四个场景。

### 9.1 单场景 smoke test

推荐最小流程：

```bash
cd semantic-gaussians

# 1) train RGB gaussian
python train.py \
  scene.scene_path=../data/lerf_ovs/teatime \
  scene.colmap_images=images \
  scene.test_cameras=False \
  train.exp_name=lerf_ovs/teatime

# 2) fusion
python fusion.py \
  scene.scene_path=../data/lerf_ovs/teatime \
  scene.colmap_images=images \
  scene.test_cameras=False \
  model.model_dir=./output/lerf_ovs/teatime \
  fusion.out_dir=./fusion_lerf_ovs/teatime \
  fusion.model_2d=openseg \
  fusion.img_dim=[<width>,<height>]

# 3) benchmark eval
python eval_lerf_ovs.py \
  eval.scene_names='[teatime]' \
  scene.scene_path=../data/lerf_ovs \
  model.model_dir=./output/lerf_ovs \
  fusion.out_dir=./fusion_lerf_ovs \
  eval.label_root=../data/lerf_ovs/label \
  eval.output_dir=./eval_result_lerf_ovs \
  eval.model_2d=openseg
```

### 9.2 扩展到四个场景

确认 `teatime` 跑通后，再统一跑：

- `figurines`
- `ramen`
- `teatime`
- `waldo_kitchen`

---

## 9.3 一键跑四个 scene 的脚本

本仓库新增了一键脚本：

```text
semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
```

它会按顺序对 4 个场景执行：

1. `train.py`
2. `fusion.py`
3. `eval_lerf_ovs.py`

默认 scene 顺序为：

- `figurines`
- `ramen`
- `teatime`
- `waldo_kitchen`

### 默认用法

```bash
bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
```

### 常用变体

只跑评测，不重跑 train / fusion：

```bash
RUN_TRAIN=0 RUN_FUSION=0 bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
```

把 3DGS 训练先缩短成 smoke test：

```bash
TRAIN_ITERS=7000 bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
```

改成 `lseg`：

```bash
MODEL_2D=lseg bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
```

### 可覆盖环境变量

- `DATA_ROOT`
- `LABEL_ROOT`
- `OUTPUT_ROOT`
- `FUSION_ROOT`
- `EVAL_ROOT`
- `MODEL_2D`
- `TRAIN_ITERS`
- `MASK_THRESH`
- `RUN_TRAIN`
- `RUN_FUSION`
- `RUN_EVAL`
- `EXTRA_TRAIN_ARGS`
- `EXTRA_FUSION_ARGS`
- `EXTRA_EVAL_ARGS`

脚本会自动读取每个 scene 的首张图像分辨率，并把它传给 `fusion.img_dim`。

---

## 10. 常见报错与排查

### 10.1 缺少 `data/lerf_ovs/label`

报错现象：

```text
Missing LERF-OVS label directory
```

原因：

- 只有场景数据，没有 benchmark 标注。

处理：

- 补齐 `label/<scene>/frame_*.json` 与对应 `.jpg`

### 10.2 GT frame 和 camera 对不上

报错现象：

```text
GT frames not found in loaded cameras
```

原因：

- 标注文件名和 COLMAP / images 中的 frame 名不一致。

处理：

- 检查 `frame_00001.json` 中 `info.name`
- 检查 scene 的 `images/` 下实际文件 stem
- 两边必须一致，例如都为 `frame_00001`

### 10.3 找不到 `0.pt`

报错现象：

```text
Missing fusion feature for scene
```

原因：

- `fusion.py` 没跑过
- 或输出目录和 eval 配置不一致

处理：

- 确认 `fusion.out_dir/<scene>/0.pt` 是否存在

### 10.4 OpenSeg 环境问题

如果 `fusion.py` 依赖的 TensorFlow / OpenSeg 环境不稳定，可以：

- 改用 `lseg`

但要保证：

- `fusion.model_2d=lseg`
- `eval.model_2d=lseg`

两边一致。

### 10.5 显存不足

可以优先尝试：

- 只评单场景 `teatime`
- 减少同时保留的可视化中间结果
- 优先完成 `train -> fusion -> eval` 的最小链路

---

## 11. 第一版 benchmark 的边界

这份实现的第一版边界是：

- **只支持 `fusion` 输出评测**
- **不支持 `distill` / 3D semantic network benchmark**
- **不复用 LangSplat autoencoder**
- **不支持直接用 `eval_segmentation.py` 的 ScanNet 标签体系**

如果后续要扩展，可以再加：

1. `eval.source=distill`
2. 多种 2D encoder 对比（OpenSeg / LSeg / SAMCLIP / VLPart）
3. 更严格的 scene-level summary 报表
4. 与 LangSplat 数值同表对齐的实验脚本

---

## 12. 交付检查清单

在你准备真正做实验前，先确认以下四项：

- [ ] `data/lerf_ovs/<scene>/images + sparse/0` 已齐全
- [ ] `data/lerf_ovs/label/<scene>/frame_*.json` 已齐全
- [ ] `output/lerf_ovs/<scene>/point_cloud/iteration_x/point_cloud.ply` 已生成
- [ ] `fusion_lerf_ovs/<scene>/0.pt` 已生成

四项都满足后，再跑 `eval_lerf_ovs.py`。

---

## 13. 本次新增文件

本 runbook 对应的实现文件是：

- `semantic-gaussians/eval_lerf_ovs.py`
- `semantic-gaussians/config/eval_lerf_ovs.yaml`
- `docs/semantic_gaussians_lerf_ovs_eval_runbook.md`

如果你后面要把这套流程整理进论文实验部分，推荐再补一份：

- 四场景统一执行脚本
- 汇总表自动导出脚本

这样后续复现实验会更稳定。
