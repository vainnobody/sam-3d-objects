# Semantic Gaussians Sample 环境适配指南（基于 sam3d conda 环境）

本文档只解决一件事：在现有 `sam3d-objects` conda 环境上，补齐 `semantic-gaussians` 运行 sample 所需依赖。

适用前提：

- 你已经在服务器上准备好了 `sam3d-objects` 环境；
- 你已经成功运行 `scripts/download_semantic_gaussians_sample.py` 下载 sample 数据；
- 你希望继续复用同一个 conda 环境，而不是再新建一个 `semantic-gaussians` 独立环境。

---

## 1. 当前环境差异

仓库里的两套环境定义并不一致：

| 项目 | sam3d 环境 | semantic-gaussians 原始环境 |
| --- | --- | --- |
| 环境文件 | `environments/default.yml` | `semantic-gaussians/environment.yml` |
| Python | 3.11 | 3.9.18 |
| CUDA | 12.1 工具链 | 11.8 |
| PyTorch | 当前仓库按 cu121 使用 | 2.1.1 + `pytorch-cuda=11.8` |

这意味着：

- 不能直接把 `semantic-gaussians/environment.yml` 整体覆盖到 `sam3d-objects` 环境；
- 推荐做法是：保留现有 `sam3d` 环境，只增量安装 `semantic-gaussians` 依赖；
- 其中最容易出问题的不是普通 Python 包，而是 `TensorFlow`、`detectron2`、`MinkowskiEngine` 和几个 CUDA 扩展。

---

## 2. 适配策略

建议按下面的优先级处理：

1. 先补齐基础 Python 依赖；
2. 再编译 `semantic-gaussians/submodules/` 里的本地 CUDA 扩展；
3. 最后按功能需要安装高风险依赖。

不要一开始就尝试一次性安装 `semantic-gaussians/requirements.txt` 里的所有内容，否则很难定位是哪一类依赖不兼容。

---

## 3. 先激活 sam3d 环境

```bash
conda activate sam3d-objects
cd /PATH/TO/sam-3d-objects
```

先确认当前 PyTorch / CUDA 组合：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果这里已经不能正常输出，再继续安装 `semantic-gaussians` 没有意义，应该先修复 `sam3d` 自身环境。

---

## 4. 安装基础依赖

`semantic-gaussians/requirements.txt` 中真正适合优先补齐的基础包主要是这些：

- `scipy==1.11.4`
- `omegaconf`
- `imageio==2.31.4`
- `scikit-image==0.22.0`
- `opencv-python`
- `ninja`
- `viser==0.1.17`
- `pytorch-lightning==2.2.4`
- `timm==0.6.13`
- `openai/CLIP`
- `PyTorch-Encoding`

建议分两步安装。

### 4.1 先安装普通 pip 包

```bash
python -m pip install \
  scipy==1.11.4 \
  omegaconf \
  imageio==2.31.4 \
  scikit-image==0.22.0 \
  opencv-python \
  ninja \
  viser==0.1.17 \
  pytorch-lightning==2.2.4 \
  timm==0.6.13
```

### 4.2 再安装 Git 依赖

```bash
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
```

说明：

- `CLIP` 基本属于后续文本特征和若干 2D 模型的共用依赖；
- `PyTorch-Encoding` 主要给 LSeg 相关模块用；
- 这两步需要联网，离线环境请先准备 wheel 或镜像源。

---

## 5. 安装 semantic-gaussians 自带子模块

`semantic-gaussians/requirements.txt` 里还有几项不是 PyPI 包，而是仓库内本地模块：

- `submodules/rgbd-rasterization`
- `submodules/channel-rasterization`
- `submodules/simple-knn`
- `submodules/segment-anything`

建议逐个安装，便于定位报错。

```bash
python -m pip install ./semantic-gaussians/submodules/segment-anything
python -m pip install ./semantic-gaussians/submodules/simple-knn
python -m pip install ./semantic-gaussians/submodules/channel-rasterization
python -m pip install ./semantic-gaussians/submodules/rgbd-rasterization
```

如果编译扩展时报找不到 `nvcc` 或 CUDA 头文件，先补这两个环境变量再重试：

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
```

如果你的服务器 CUDA toolkit 不在 conda 环境内，而是在系统路径，也可以把 `CUDA_HOME` 指向系统 CUDA 目录。

---

## 6. 高风险依赖如何处理

下面三类依赖不建议和基础依赖一起装。

### 6.1 TensorFlow / OpenSeg

`semantic-gaussians/model/openseg_predictor.py` 直接依赖：

- `tensorflow`
- `tensorflow.compat.v1`

而 `semantic-gaussians/requirements.txt` 里给的是：

```text
tensorflow[and-cuda]==2.14.0
```

这是当前适配里最不稳定的一项，原因有两个：

- `semantic-gaussians` 原始环境是 Python 3.9 + CUDA 11.8；
- 你当前环境是 `sam3d` 的 Python 3.11 + CUDA 12.1 思路。

建议：

- 如果你当前只是做 sample 环境适配，先不要把 TensorFlow 作为阻塞项；
- 只有在你确定要走 `openseg` 路线时，再单独安装和验证 TensorFlow；
- 一旦安装 TensorFlow 后出现 CUDA 库冲突，优先考虑把 `OpenSeg` 相关功能隔离使用，而不是回滚整个 sam3d 环境。

注意：

- `semantic-gaussians/view_viser.py` 文件顶部直接 `import OpenSeg`；
- 所以只要运行这个脚本，就可能要求 TensorFlow 已经可导入，即使你最终选择的是 `lseg`。

### 6.2 detectron2 / VLPart

`semantic-gaussians/model/vlpart_predictor.py` 依赖：

- `detectron2`
- `segment_anything`

而 `detectron2` 往往与 Python、PyTorch、CUDA 版本强绑定。建议：

- 如果只是先适配 sample，不要优先安装 `detectron2`；
- 仅当你需要 `fusion.model_2d=vlpart` 时再处理它。

### 6.3 MinkowskiEngine

`semantic-gaussians/eval_segmentation.py` 和 `semantic-gaussians/model/mink_unet.py` 依赖 `MinkowskiEngine`。

这部分主要用于：

- 3D semantic distillation；
- segmentation evaluation。

如果你现在目标只是把 sample 环境补齐，不一定需要马上装它。建议把它放到最后一步。

官方 README 给的是源码编译方式，仍然可以沿用，但要接受它对编译器、CUDA、BLAS 都比较敏感。

---

## 7. 建议的最小验证顺序

### 7.1 基础依赖验证

```bash
python -c "import torch, cv2, imageio, omegaconf, timm, clip, pytorch_lightning, viser; print('base deps ok')"
```

### 7.2 本地子模块验证

```bash
python -c "import segment_anything; print('segment_anything ok')"
python -c "import simple_knn._C; print('simple_knn ok')"
python -c "import channel_rasterization._C; print('channel_rasterization ok')"
python -c "import rgbd_rasterization._C; print('rgbd_rasterization ok')"
```

### 7.3 可选依赖按需验证

如果你装了 TensorFlow：

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

如果你装了 detectron2：

```bash
python -c "import detectron2; print('detectron2 ok')"
```

如果你装了 MinkowskiEngine：

```bash
python -c "import MinkowskiEngine as ME; print(ME.__version__)"
```

---

## 8. 和 sample 数据的关系

你已经跑过：

```bash
python scripts/download_semantic_gaussians_sample.py
```

该脚本会把 Mip-NeRF 360 sample 数据整理到类似下面的结构：

```text
data/semantic_gaussians_samples/<scene_name>/
  images_*/
  sparse_*/0/
```

这类目录结构与 `semantic-gaussians/config/fusion_mipnerf360.yaml` 里期待的 COLMAP 风格数据是对齐的，因此“先补环境，再改 config”是合理顺序。

但这份文档只负责环境适配，不展开配置文件修改和训练/融合命令。

---

## 9. 推荐安装顺序总结

实际操作时，建议严格按下面顺序：

```bash
conda activate sam3d-objects

# 1) 基础依赖
python -m pip install \
  scipy==1.11.4 omegaconf imageio==2.31.4 scikit-image==0.22.0 \
  opencv-python ninja viser==0.1.17 pytorch-lightning==2.2.4 timm==0.6.13

# 2) Git 依赖
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/

# 3) 本地模块
python -m pip install ./semantic-gaussians/submodules/segment-anything
python -m pip install ./semantic-gaussians/submodules/simple-knn
python -m pip install ./semantic-gaussians/submodules/channel-rasterization
python -m pip install ./semantic-gaussians/submodules/rgbd-rasterization
```

然后先做 import 验证，确认无误后，再决定是否继续安装：

- TensorFlow / OpenSeg
- detectron2 / VLPart
- MinkowskiEngine

---

## 10. 常见问题

### 10.1 `nvcc: command not found`

说明当前 shell 没有拿到 CUDA toolkit。优先检查：

```bash
echo $CUDA_HOME
which nvcc
```

若 `sam3d` 环境内自带 CUDA toolkit，可执行：

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
```

### 10.2 `No module named 'simple_knn._C'`

通常表示 `simple-knn` 没有成功编译，而不是 Python 没装好。直接回到子模块安装步骤重装：

```bash
python -m pip install --force-reinstall ./semantic-gaussians/submodules/simple-knn
```

### 10.3 `view_viser.py` 一启动就报 TensorFlow 相关错误

这是因为 `view_viser.py` 顶部直接导入了 `OpenSeg`。即使你想走 `lseg`，脚本 import 阶段也可能先触发 TensorFlow 依赖。

所以：

- 如果你要用 `view_viser.py`，默认要把 TensorFlow 这条链路也配好；
- 如果你当前只做 sample 环境适配，可以先不把 viewer 当作第一阶段验证目标。

### 10.4 `detectron2` 或 `MinkowskiEngine` 编译失败

这两项都高度依赖当前服务器的：

- Python 版本；
- PyTorch 版本；
- CUDA 版本；
- 编译器版本。

如果基础 sample 环境已经能跑，不建议为了这两个可选模块去大改现有 `sam3d` 环境。

---

## 11. 一句话结论

对当前仓库来说，最稳妥的适配方式不是“把 `semantic-gaussians` 原环境完整复制进来”，而是：

- 继续使用 `sam3d-objects` conda 环境；
- 先安装基础依赖和本地 CUDA 子模块；
- 把 TensorFlow、detectron2、MinkowskiEngine 视为按需追加的功能依赖。
