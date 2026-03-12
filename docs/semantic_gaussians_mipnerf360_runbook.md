# 在 360 数据集上跑通 Semantic Gaussians

本文档面向已经完成环境适配的情况，目标是在 `Mip-NeRF 360` / COLMAP 风格数据上跑通 `semantic-gaussians` 的最小链路。

默认假设：

- 你已经准备好 `sam3d-objects` conda 环境；
- 你已经按 `docs/semantic_gaussians_sample_env_on_sam3d.md` 补齐了基础依赖；
- 你准备运行的是当前仓库里的 `semantic-gaussians/` 子项目；
- 你手头的数据是 `Mip-NeRF 360` 风格的 COLMAP 场景，或者由 `scripts/download_semantic_gaussians_sample.py` 下载出的 sample。

本文重点覆盖：

1. 准备 360 数据目录；
2. 训练 RGB Gaussian；
3. 在高斯上做 2D 语义投影；
4. 可选查看结果。

不覆盖：

- ScanNet 语义评测；
- 3D semantic distillation 指标复现；
- 论文级别完整训练配置搜索。

---

## 1. 先理解这条链路要跑什么

对 360 数据集，最小可跑通链路是：

```text
COLMAP scene -> train.py -> RGB Gaussian checkpoint
            -> fusion.py -> fused semantic feature (.pt)
            -> view_viser.py (optional)
```

这里有两个关键点：

- `train.py` 负责先训练普通 RGB 版 3D Gaussian；
- `fusion.py` 再把 2D 模型特征投影到训练好的 Gaussian 上。

如果你只是想验证 `semantic-gaussians` 在 360 数据上能跑通，跑到 `fusion.py` 产出 `.pt` 文件就已经算打通主链路。

---

## 2. 数据目录要求

`semantic-gaussians/scene/scene.py` 对 COLMAP 数据的判断非常直接：场景目录下必须存在 `sparse/`。

标准可识别结构如下：

```text
<scene_root>/
  images/ 或 images_2 / images_4
  sparse/
    0/
      cameras.bin
      images.bin
      points3D.bin
```

### 2.1 如果你使用的是本仓库 sample 下载脚本

`scripts/download_semantic_gaussians_sample.py` 下载出来的 sample 通常是这种结构：

```text
data/semantic_gaussians_samples/stump/
  images_4/
  sparse_4/
    0/
```

注意：

- `images_4/` 可以通过配置项 `scene.colmap_images=images_4` 指定；
- 但 `scene/scene.py` 只会检查 `sparse/`，不会自动识别 `sparse_4/`。

所以 sample 数据要先做一个软链接：

```bash
cd /PATH/TO/sam-3d-objects/data/semantic_gaussians_samples/stump
ln -sfn sparse_4 sparse
```

如果你下载的是 `bonsai`，对应改成：

```bash
ln -sfn sparse_2 sparse
```

### 2.2 如果你用的是完整 Mip-NeRF 360 场景

只要目录中已经有：

- `images/` 或 `images_*`
- `sparse/0/cameras.bin`
- `sparse/0/images.bin`
- `sparse/0/points3D.bin`

就可以直接用，不需要额外预处理。

---

## 3. 建议从 `semantic-gaussians/` 目录运行

这一步很重要。

`train.py`、`fusion.py`、`view_viser.py` 都是用相对路径加载 `./config/*.yaml`，所以推荐统一这样进入：

```bash
conda activate sam3d-objects
cd /PATH/TO/sam-3d-objects/semantic-gaussians
```

后文命令都默认你当前就在 `semantic-gaussians/` 目录下。

---

## 4. 准备 2D 模型权重

对 360 数据，建议优先用 `LSeg` 跑通，而不是 `OpenSeg`。

原因：

- `fusion.py` 支持 `openseg / lseg / samclip / vlpart`；
- `OpenSeg` 依赖 TensorFlow，环境冲突概率更高；
- `LSeg` 对当前“先跑通”更友好。

### 4.1 放置 LSeg 权重

`fusion.py` 在 `model_2d=lseg` 时会默认读取：

```text
./weights/lseg/demo_e200.ckpt
```

所以需要先准备好这个文件：

```text
semantic-gaussians/
  weights/
    lseg/
      demo_e200.ckpt
```

`semantic-gaussians/model/lseg/README.MD` 里也写的是这个 demo checkpoint 命名。

---

## 5. 先检查图像分辨率

`fusion.py` 里的 `fusion.img_dim` 必须和场景实际图像大小匹配。

可以先随便读一张图确认宽高：

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image

img_dir = Path("../data/semantic_gaussians_samples/stump/images_4")
img = sorted(img_dir.iterdir())[0]
print(img)
print(Image.open(img).size)  # (width, height)
PY
```

记下输出的 `(width, height)`，后面要填给：

- `fusion.fusion.img_dim=[width,height]`

不要直接照搬 `fusion_mipnerf360.yaml` 里的默认值，除非你确认图像大小完全一致。

---

## 6. 第一步：训练 RGB Gaussian

先训练 360 场景的普通 RGB Gaussian。

下面给一个 sample 场景 `stump` 的最小示例：

```bash
python train.py \
  scene.scene_path=../data/semantic_gaussians_samples/stump \
  scene.colmap_images=images_4 \
  scene.test_cameras=False \
  train.exp_name=stump_rgbgs \
  train.iterations=7000 \
  train.test_iterations='[100,7000]' \
  train.save_iterations='[7000]'
```

说明：

- `scene.scene_path` 指向场景根目录；
- `scene.colmap_images=images_4` 告诉 loader 用下采样图像；
- `scene.test_cameras=False` 对 sample 跑通更省事；
- `train.exp_name=stump_rgbgs` 最终会把模型写到 `./output/stump_rgbgs/`；
- `7000` 是一个适合 smoke test 的值，先验证流程通不通。

如果你要追求更完整的渲染质量，可以后续再拉到默认的 `30000` iteration。

### 6.1 训练产物位置

训练完成后，主要关注：

```text
semantic-gaussians/output/stump_rgbgs/
  config.yaml
  point_cloud/
    iteration_7000/
      point_cloud.ply
```

如果这里已经有 `point_cloud.ply`，说明第一步打通了。

---

## 7. 第二步：做 2D 语义投影

接着把 2D 模型特征投影到高斯上。

推荐仍然先用 `LSeg`。

```bash
python fusion.py \
  scene.scene_path=../data/semantic_gaussians_samples/stump \
  scene.colmap_images=images_4 \
  scene.test_cameras=False \
  model.model_dir=./output/stump_rgbgs \
  model.load_iteration=7000 \
  fusion.model_2d=lseg \
  fusion.img_dim='[WIDTH,HEIGHT]' \
  fusion.out_dir=./fusion/stump_lseg
```

把 `WIDTH` 和 `HEIGHT` 替换成上一步查到的真实图像宽高。

示例，如果图像大小是 `779x519`：

```bash
python fusion.py \
  scene.scene_path=../data/semantic_gaussians_samples/stump \
  scene.colmap_images=images_4 \
  scene.test_cameras=False \
  model.model_dir=./output/stump_rgbgs \
  model.load_iteration=7000 \
  fusion.model_2d=lseg \
  fusion.img_dim='[779,519]' \
  fusion.out_dir=./fusion/stump_lseg
```

### 7.1 fusion 成功后会产出什么

正常情况下，你会得到：

```text
semantic-gaussians/fusion/stump_lseg/
  0.pt
```

这个 `.pt` 文件就是后续 viewer 或其他分析用的 fused semantic feature。

如果你已经拿到了这个文件，就说明 360 数据上的 `train -> fusion` 主链路已经跑通。

---

## 8. 第三步：可选查看语义结果

理论上可以用：

```bash
python view_viser.py \
  scene.scene_path=../data/semantic_gaussians_samples/stump \
  scene.colmap_images=images_4 \
  scene.test_cameras=False \
  model.model_dir=./output/stump_rgbgs \
  model.load_iteration=7000 \
  render.fusion_dir=./fusion/stump_lseg/0.pt \
  render.model_2d=lseg
```

但这里要特别注意：

- `view_viser.py` 文件顶部直接导入了 `OpenSeg`；
- 所以即使你命令里写的是 `render.model_2d=lseg`，脚本 import 阶段也可能先要求 TensorFlow 可导入。

因此：

- 如果你的环境里还没装好 TensorFlow / OpenSeg，这一步可能直接失败；
- 对“先跑通 360 数据 SG 主链路”来说，viewer 不是第一优先级；
- 建议先把 `train.py` 和 `fusion.py` 跑通，再决定要不要为 viewer 补 TensorFlow。

---

## 9. 推荐的最小验收标准

满足下面 3 条，就可以认为你已经在 360 数据集上跑通了当前仓库的 SG 主流程：

1. `train.py` 成功输出 `point_cloud/iteration_xxx/point_cloud.ply`
2. `fusion.py` 成功输出 `fusion/.../0.pt`
3. `fusion.py` 过程中没有出现 2D 模型权重缺失或 CUDA 扩展导入失败

---

## 10. 常见问题

### 10.1 报错 `Could not recognize scene type!`

优先检查场景目录下是否真的有：

- `sparse/`
- 或 `transforms_train.json`
- 或 `pose/`

对 360/COLMAP 数据，最常见原因就是你只有 `sparse_4/`，没有 `sparse/`。

解决方法：

```bash
ln -sfn sparse_4 sparse
```

### 10.2 报错找不到 LSeg 权重

确认路径必须是：

```text
semantic-gaussians/weights/lseg/demo_e200.ckpt
```

因为 `fusion.py` 的 `lseg` 分支写死了这个默认位置。

### 10.3 `fusion.py` 跑起来后尺寸不匹配

通常是 `fusion.img_dim` 配错了。

先重新检查：

- 你实际使用的是 `images`、`images_2` 还是 `images_4`；
- `fusion.img_dim` 是否与该目录下图片的 `(width, height)` 完全一致。

### 10.4 `view_viser.py` 要求 TensorFlow

这是当前代码结构导致的，不是你命令写错了。

如果你只是先验证 360 数据上的 SG 链路，请先把验收目标定在：

- `train.py` 成功；
- `fusion.py` 成功。

---

## 11. 一套可直接改路径的命令模板

```bash
conda activate sam3d-objects
cd /PATH/TO/sam-3d-objects/semantic-gaussians

# sample 数据若只有 sparse_4 / sparse_2，先补 sparse 软链接
cd ../data/semantic_gaussians_samples/stump
ln -sfn sparse_4 sparse
cd ../../semantic-gaussians

# 训练 RGB Gaussian
python train.py \
  scene.scene_path=../data/semantic_gaussians_samples/stump \
  scene.colmap_images=images_4 \
  scene.test_cameras=False \
  train.exp_name=stump_rgbgs \
  train.iterations=7000 \
  train.test_iterations='[100,7000]' \
  train.save_iterations='[7000]'

# 语义投影（把 WIDTH,HEIGHT 改成真实尺寸）
python fusion.py \
  scene.scene_path=../data/semantic_gaussians_samples/stump \
  scene.colmap_images=images_4 \
  scene.test_cameras=False \
  model.model_dir=./output/stump_rgbgs \
  model.load_iteration=7000 \
  fusion.model_2d=lseg \
  fusion.img_dim='[WIDTH,HEIGHT]' \
  fusion.out_dir=./fusion/stump_lseg
```

如果这两步都成功，说明当前仓库已经可以在 360 数据上跑通 `Semantic Gaussians` 的最小流程。
