# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Stage 1 Only Inference - 只加载稀疏结构生成相关模型

本模块提供了一个轻量级的推理类，只加载 Stage 1（稀疏结构生成）相关的模型参数，
适用于只需要生成大致几何形状的场景，可以显著减少内存占用。

主要用途：
- 快速预览几何形状
- 稀疏点云提取
- 形状潜在特征分析

使用示例：
    from stage1_inference import Stage1OnlyInference, load_image, load_single_mask
    
    inference = Stage1OnlyInference(config_path)
    output = inference.run(image, mask, steps=4, use_distillation=True)
    
    # 获取稀疏点云坐标
    voxel = output["voxel"]  # (N, 3) 归一化坐标
    
    # 获取形状潜在特征
    shape = output["shape"]  # (B, 8, 16, 16, 16)
"""

import os

# not ideal to put that here
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
from typing import Union, Optional, List, Callable
from functools import wraps
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
import torch
from torch.utils._pytree import tree_map_only
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import builtins
import shutil
import subprocess

import sam3d_objects  # do not remove this import
from sam3d_objects.pipeline import preprocess_utils
from sam3d_objects.pipeline.inference_utils import (
    get_pose_decoder,
    downsample_sparse_structure,
    prune_sparse_structure,
)
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from loguru import logger

__all__ = [
    "Stage1OnlyInference",
    "load_image",
    "load_single_mask",
    "load_masks",
    "display_image",
    "visualize_sparse_coords",
    "visualize_shape_features",
]

# Safety filters for Hydra instantiation
WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]


def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


class Stage1OnlyInference:
    """
    只加载 Stage 1（稀疏结构生成）相关模型的推理类。
    
    相比完整的 Inference 类，本类只加载以下模型：
    - ss_generator: 稀疏结构生成器（Flow Matching）
    - ss_decoder: VAE 解码器（潜在表示 → 64³体素网格）
    - ss_condition_embedder: DINO 条件嵌入器
    - ss_preprocessor: 图像预处理器
    - pose_decoder: 姿态解码器
    
    不加载以下 Stage 2 模型：
    - slat_generator: 结构化潜在生成器
    - slat_decoder_gs/mesh: Gaussian/网格解码器
    - slat_condition_embedder: Stage 2 条件嵌入器
    
    这可以显著减少内存占用（约减少 50% 显存）。
    """
    
    def __init__(self, config_file: str, compile: bool = False):
        """
        初始化 Stage 1 推理管道。
        
        Args:
            config_file: pipeline.yaml 配置文件路径
            compile: 是否使用 torch.compile 加速（实验性）
        """
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        
        # 提取 Stage 1 相关配置
        self._init_from_config(config)
        
    def _init_from_config(self, config):
        """从配置初始化 Stage 1 模型"""
        self.device = torch.device(config.get("device", "cuda"))
        self.compile_model = config.get("compile_model", False)
        self.workspace_dir = config.get("workspace_dir", "")
        
        # Stage 1 推理参数
        self.ss_inference_steps = config.get("ss_inference_steps", 25)
        self.ss_rescale_t = config.get("ss_rescale_t", 3)
        self.ss_cfg_strength = config.get("ss_cfg_strength", 7)
        self.ss_cfg_interval = config.get("ss_cfg_interval", [0, 500])
        self.ss_cfg_strength_pm = config.get("ss_cfg_strength_pm", 0.0)
        self.ss_condition_input_mapping = config.get("ss_condition_input_mapping", ["image"])
        self.pad_size = config.get("pad_size", 1.0)
        
        # 数据类型
        self.dtype = self._get_dtype(config.get("dtype", "bfloat16"))
        shape_model_dtype = config.get("shape_model_dtype")
        self.shape_model_dtype = self._get_dtype(shape_model_dtype) if shape_model_dtype else self.dtype
        
        # 初始化 Stage 1 模型
        logger.info("Loading Stage 1 models only...")
        
        with self.device:
            # 1. Pose decoder
            pose_decoder_name = config.get("pose_decoder_name", "default")
            self.pose_decoder = self._init_pose_decoder(
                config.ss_generator_config_path, pose_decoder_name
            )
            
            # 2. Preprocessor - 始终使用默认预处理器
            self.ss_preprocessor = preprocess_utils.get_default_preprocessor()
            
            # 3. SS Generator
            ss_generator = self._init_ss_generator(
                config.ss_generator_config_path, 
                config.ss_generator_ckpt_path
            )
            
            # 4. SS Decoder
            ss_decoder = self._init_ss_decoder(
                config.ss_decoder_config_path,
                config.ss_decoder_ckpt_path
            )
            
            # 5. Condition embedder
            ss_condition_embedder = self._init_ss_condition_embedder(
                config.ss_generator_config_path,
                config.ss_generator_ckpt_path
            )
            
            # Override CFG config
            self._override_ss_generator_cfg_config(
                ss_generator,
                cfg_strength=self.ss_cfg_strength,
                inference_steps=self.ss_inference_steps,
                rescale_t=self.ss_rescale_t,
                cfg_interval=self.ss_cfg_interval,
                cfg_strength_pm=self.ss_cfg_strength_pm,
            )
            
            # Store models
            self.models = torch.nn.ModuleDict({
                "ss_generator": ss_generator,
                "ss_decoder": ss_decoder,
            })
            
            self.condition_embedders = {
                "ss_condition_embedder": ss_condition_embedder,
            }
            
        logger.info("Stage 1 models loaded successfully!")
        
        if self.compile_model:
            self._compile_models()
    
    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """将字符串转换为 torch.dtype"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def _init_pose_decoder(self, ss_generator_config_path, pose_decoder_name):
        """初始化姿态解码器"""
        if pose_decoder_name is None:
            config = OmegaConf.load(
                os.path.join(self.workspace_dir, ss_generator_config_path)
            )
            if "module" in config and "pose_target_convention" in config["module"]:
                pose_decoder_name = config["module"]["pose_target_convention"]
            else:
                pose_decoder_name = "default"
        logger.info(f"Using pose decoder: {pose_decoder_name}")
        return get_pose_decoder(pose_decoder_name)
    
    def _init_ss_preprocessor(self, ss_preprocessor, ss_generator_config_path):
        """初始化图像预处理器"""
        if ss_preprocessor is not None:
            return ss_preprocessor
        
        # 尝试从配置加载
        try:
            config = OmegaConf.load(
                os.path.join(self.workspace_dir, ss_generator_config_path)
            )
            if "tdfy" in config and "val_preprocessor" in config["tdfy"]:
                return instantiate(config["tdfy"]["val_preprocessor"])
        except Exception as e:
            logger.warning(f"Failed to load preprocessor from config: {e}")
        
        # 使用默认预处理器
        logger.info("Using default preprocessor")
        return preprocess_utils.get_default_preprocessor()
    
    def _init_ss_generator(self, config_path, ckpt_path):
        """初始化稀疏结构生成器"""
        config = OmegaConf.load(os.path.join(self.workspace_dir, config_path))
        model_config = config["module"]["generator"]["backbone"]
        model = instantiate(model_config)
        
        # Load checkpoint
        ckpt_path = os.path.join(self.workspace_dir, ckpt_path)
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )
        
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
            state_dict = state_dict_prefix_func(state_dict)
            model.load_state_dict(state_dict, strict=False)
        else:
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=True,
                device="cpu",
                freeze=True,
                eval=True,
                state_dict_key="state_dict",
                state_dict_fn=state_dict_prefix_func,
            )
        
        model = model.to(self.device).to(self.shape_model_dtype)
        model.eval()
        
        return model
    
    def _init_ss_decoder(self, config_path, ckpt_path):
        """初始化 VAE 解码器"""
        config = OmegaConf.load(os.path.join(self.workspace_dir, config_path))
        # 删除可能存在的 pretrained_ckpt_path 避免加载问题
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        
        model = instantiate(config)
        
        # Load checkpoint
        ckpt_path = os.path.join(self.workspace_dir, ckpt_path)
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
            model.load_state_dict(state_dict, strict=False)
        else:
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=True,
                device="cpu",
                freeze=True,
                eval=True,
                state_dict_key=None,  # 注意这里是 None
            )
        
        model = model.to(self.device).to(self.shape_model_dtype)
        model.eval()
        
        return model
    
    def _init_ss_condition_embedder(self, config_path, ckpt_path):
        """初始化 DINO 条件嵌入器"""
        config = OmegaConf.load(os.path.join(self.workspace_dir, config_path))
        
        # 检查是否有 condition_embedder 配置
        if "condition_embedder" not in config.get("module", {}):
            logger.info("No condition_embedder found in config, returning None")
            return None
        
        model_config = config["module"]["condition_embedder"]["backbone"]
        model = instantiate(model_config)
        
        # Load checkpoint
        ckpt_path = os.path.join(self.workspace_dir, ckpt_path)
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.condition_embedder."
        )
        
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
            state_dict = state_dict_prefix_func(state_dict)
            model.load_state_dict(state_dict, strict=False)
        else:
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=True,
                device="cpu",
                freeze=True,
                eval=True,
                state_dict_key="state_dict",
                state_dict_fn=state_dict_prefix_func,
            )
        
        model = model.to(self.device).to(self.shape_model_dtype)
        model.eval()
        
        return model
    
    def _override_ss_generator_cfg_config(
        self,
        ss_generator,
        cfg_strength=7,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
        cfg_strength_pm=0.0,
    ):
        """配置 CFG 参数"""
        ss_generator.inference_steps = inference_steps
        ss_generator.reverse_fn.strength = cfg_strength
        ss_generator.reverse_fn.interval = cfg_interval
        ss_generator.rescale_t = rescale_t
        ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
        ss_generator.reverse_fn.unconditional_handling = "add_flag"
        ss_generator.reverse_fn.strength_pm = cfg_strength_pm
        
        logger.info(
            f"SS Generator config: steps={inference_steps}, cfg={cfg_strength}, "
            f"interval={cfg_interval}, rescale_t={rescale_t}, cfg_pm={cfg_strength_pm}"
        )
    
    def _compile_models(self):
        """使用 torch.compile 编译模型"""
        logger.info("Compiling models with torch.compile...")
        
        def clone_output_wrapper(fn):
            @wraps(fn)
            def wrapped(*args, **kwargs):
                return tree_map_only(torch.Tensor, lambda x: x.clone(), fn(*args, **kwargs))
            return wrapped
        
        self.models["ss_generator"].reverse_fn.inner_forward = clone_output_wrapper(
            self.models["ss_generator"].reverse_fn.inner_forward,
        )
        self.models["ss_decoder"].forward = clone_output_wrapper(
            self.models["ss_decoder"].forward,
        )
        
        self.models["ss_generator"].reverse_fn.inner_forward = torch.compile(
            self.models["ss_generator"].reverse_fn.inner_forward,
            mode="reduce-overhead",
            fullgraph=False,
        )
        self.models["ss_decoder"].forward = torch.compile(
            self.models["ss_decoder"].forward,
            mode="reduce-overhead",
            fullgraph=False,
        )
        
        logger.info("Model compilation complete!")
    
    def merge_mask_to_rgba(self, image, mask):
        """将掩码合并到图像的 alpha 通道"""
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image
    
    def is_mm_dit(self):
        """检查是否是多模态 DiT"""
        return hasattr(
            self.models["ss_generator"].reverse_fn.backbone, 
            "latent_mapping"
        )
    
    def embed_condition(self, condition_embedder, *args, **kwargs):
        """嵌入条件"""
        if condition_embedder is not None:
            tokens = condition_embedder(*args, **kwargs)
            return tokens, None, None
        return None, args, kwargs
    
    def get_condition_input(self, condition_embedder, input_dict, input_mapping):
        """获取条件输入"""
        condition_args = [input_dict[k] for k in input_mapping]
        condition_kwargs = {
            k: v for k, v in input_dict.items() if k not in input_mapping
        }
        logger.info("Running condition embedder...")
        embedded_cond, condition_args, condition_kwargs = self.embed_condition(
            condition_embedder, *condition_args, **condition_kwargs
        )
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}
        return condition_args, condition_kwargs
    
    # ============== 预处理方法 ==============
    
    def image_to_float(self, image):
        """将图像转换为浮点数"""
        image = np.array(image)
        image = image / 255
        image = image.astype(np.float32)
        return image
    
    def _apply_transform(self, input: torch.Tensor, transform):
        """应用变换"""
        if input is not None and transform is not None and transform != (None,):
            input = transform(input)
        return input
    
    def _preprocess_image_and_mask(self, rgb_image, mask_image, img_mask_joint_transform):
        """预处理图像和掩码"""
        if img_mask_joint_transform is not None and img_mask_joint_transform != (None,):
            for trans in img_mask_joint_transform:
                rgb_image, mask_image = trans(rgb_image, mask_image)
        return rgb_image, mask_image
    
    def preprocess_image(self, image, preprocessor):
        """预处理图像"""
        # canonical type is numpy
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = (get_mask(rgba_image, None, "ALPHA_CHANNEL") > 0).float()
        processed_rgb_image, processed_mask = self._preprocess_image_and_mask(
            rgb_image, rgb_image_mask, preprocessor.img_mask_joint_transform
        )

        # transform tensor to model input
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, preprocessor.img_transform
        )
        processed_mask = self._apply_transform(
            processed_mask, preprocessor.mask_transform
        )

        # full image, with only processing from the image
        rgb_image = self._apply_transform(rgb_image, preprocessor.img_transform)
        rgb_image_mask = self._apply_transform(
            rgb_image_mask, preprocessor.mask_transform
        )
        item = {
            "mask": processed_mask[None].to(self.device),
            "image": processed_rgb_image[None].to(self.device),
            "rgb_image": rgb_image[None].to(self.device),
            "rgb_image_mask": rgb_image_mask[None].to(self.device),
        }

        return item
    
    def sample_sparse_structure(
        self, 
        ss_input_dict: dict, 
        inference_steps: int = None, 
        use_distillation: bool = False
    ):
        """
        采样稀疏结构。
        
        Args:
            ss_input_dict: 包含 'image' 键的输入字典
            inference_steps: 推理步数（None 则使用默认值）
            use_distillation: 是否使用蒸馏加速模式
        
        Returns:
            dict: 包含 'coords', 'shape', 'scale', 'rotation', 'translation' 等
        """
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        
        # 配置蒸馏模式
        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm
        
        # 保存并设置推理步数
        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps
        
        image = ss_input_dict["image"]
        bs = image.shape[0]
        
        logger.info(
            f"Sampling sparse structure: steps={ss_generator.inference_steps}, "
            f"strength={ss_generator.reverse_fn.strength}, "
            f"distillation={use_distillation}"
        )
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                # 确定潜在形状
                if self.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)
                
                # 获取条件嵌入
                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["ss_condition_embedder"],
                    ss_input_dict,
                    self.ss_condition_input_mapping,
                )
                
                # 运行生成器
                return_dict = ss_generator(
                    latent_shape_dict,
                    image.device,
                    *condition_args,
                    **condition_kwargs,
                )
                if not self.is_mm_dit():
                    return_dict = {"shape": return_dict}
                
                # 解码
                shape_latent = return_dict["shape"]
                ss = ss_decoder(
                    shape_latent.permute(0, 2, 1).contiguous()
                )
                
                # 处理解码输出
                return_dict["coords"] = ss.coords
                return_dict["shape"] = shape_latent.permute(0, 2, 1)  # (B, 8, 16, 16, 16)
                
                # 下采样和裁剪
                if self.pad_size < 1.0:
                    return_dict = prune_sparse_structure(return_dict)
                if ss_input_dict.get("downsample_ss_dist", 0) > 0:
                    return_dict = downsample_sparse_structure(
                        return_dict, 
                        ss_input_dict["downsample_ss_dist"]
                    )
                
                # 计算下采样因子
                orig_coords = return_dict["coords_original"] if "coords_original" in return_dict else return_dict["coords"]
                downsample_factor = orig_coords.shape[0] / return_dict["coords"].shape[0]
                return_dict["downsample_factor"] = downsample_factor
        
        # 恢复推理步数
        ss_generator.inference_steps = prev_inference_steps
        
        return return_dict
    
    def run(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]] = None,
        seed: Optional[int] = None,
        steps: int = 4,
        use_distillation: bool = True,
    ) -> dict:
        """
        运行 Stage 1 推理。
        
        Args:
            image: 输入图像 (H, W, 3) 或 PIL Image
            mask: 二值掩码 (H, W)，可选
            seed: 随机种子
            steps: 推理步数（默认 4，用于蒸馏模式）
            use_distillation: 是否使用蒸馏加速（默认 True）
        
        Returns:
            dict: 包含以下键：
                - coords: (N, 4) 稀疏体素坐标 [batch, x, y, z]
                - voxel: (N, 3) 归一化坐标，范围 [-0.5, 0.5]
                - shape: (B, 8, 16, 16, 16) 形状潜在特征
                - scale: (1, 3) 物体尺度
                - rotation: (1, 4) 四元数旋转
                - translation: (1, 3) 平移向量
        """
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
        
        # 合并掩码
        if mask is not None:
            rgba_image = self.merge_mask_to_rgba(image, mask)
        else:
            rgba_image = image
        
        # 预处理
        ss_input_dict = self.preprocess_image(rgba_image, self.ss_preprocessor)
        
        # 运行稀疏结构采样
        ss_return_dict = self.sample_sparse_structure(
            ss_input_dict,
            inference_steps=steps,
            use_distillation=use_distillation,
        )
        
        # 姿态解码
        ss_return_dict.update(self.pose_decoder(ss_return_dict))
        
        # 缩放
        if "scale" in ss_return_dict:
            ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict.get("downsample_factor", 1.0)
        
        # 计算归一化体素坐标
        ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
        
        logger.info(f"Stage 1 inference complete! Points: {ss_return_dict['coords'].shape[0]}")
        
        return ss_return_dict


# ============== 辅助函数 ==============

def load_image(path: str) -> np.ndarray:
    """加载图像"""
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path: str) -> np.ndarray:
    """加载掩码"""
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask


def load_single_mask(folder_path: str, index: int = 0, extension: str = ".png") -> np.ndarray:
    """加载单个掩码"""
    masks = load_masks(folder_path, [index], extension)
    return masks[0]


def load_masks(folder_path: str, indices_list: List[int] = None, extension: str = ".png") -> List[np.ndarray]:
    """加载多个掩码"""
    masks = []
    indices_list = [] if indices_list is None else list(indices_list)
    
    if not len(indices_list) > 0:
        idx = 0
        while os.path.exists(os.path.join(folder_path, f"{idx}{extension}")):
            indices_list.append(idx)
            idx += 1
    
    for idx in indices_list:
        mask_path = os.path.join(folder_path, f"{idx}{extension}")
        assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
        mask = load_mask(mask_path)
        masks.append(mask)
    
    return masks


def display_image(image, masks=None):
    """显示图像和掩码"""
    import seaborn as sns
    
    def imshow(image, ax):
        ax.axis("off")
        ax.imshow(image)
    
    grid = (1, 1) if masks is None else (2, 2)
    fig, axes = plt.subplots(*grid)
    
    if masks is not None:
        mask_colors = sns.color_palette("husl", len(masks))
        black_image = np.zeros_like(image[..., :3], dtype=float)
        mask_display = np.copy(black_image)
        mask_union = np.zeros_like(image[..., :3])
        
        for i, mask in enumerate(masks):
            mask_display[mask] = mask_colors[i]
            mask_union |= mask[..., None] if mask.ndim == 2 else mask
        
        imshow(black_image, axes[0, 1])
        imshow(mask_display, axes[1, 0])
        imshow(image * mask_union, axes[1, 1])
    
    image_axe = axes if masks is None else axes[0, 0]
    imshow(image, image_axe)
    
    fig.tight_layout(pad=0)
    fig.show()


# ============== 可视化函数 ==============

def visualize_sparse_coords(
    voxel: np.ndarray,
    title: str = "Sparse Point Cloud",
    color: str = "blue",
    alpha: float = 0.6,
    s: int = 1,
    figsize: tuple = (10, 8),
    ax = None,
):
    """
    可视化稀疏点云坐标。
    
    Args:
        voxel: (N, 3) 归一化坐标数组
        title: 图标题
        color: 点颜色
        alpha: 透明度
        s: 点大小
        figsize: 图像尺寸
        ax: 已有的 3D axes（可选）
    
    Returns:
        matplotlib Figure 或 Axes
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False
    
    ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2], c=color, alpha=alpha, s=s)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置等比例
    max_range = np.array([
        voxel[:, 0].max() - voxel[:, 0].min(),
        voxel[:, 1].max() - voxel[:, 1].min(),
        voxel[:, 2].max() - voxel[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (voxel[:, 0].max() + voxel[:, 0].min()) * 0.5
    mid_y = (voxel[:, 1].max() + voxel[:, 1].min()) * 0.5
    mid_z = (voxel[:, 2].max() + voxel[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if return_fig:
        return fig
    return ax


def visualize_shape_features(
    shape: np.ndarray,
    method: str = "pca",
    title: str = "Shape Latent Features",
    figsize: tuple = (12, 5),
):
    """
    可视化形状潜在特征。
    
    Args:
        shape: (B, 8, 16, 16, 16) 形状潜在特征数组
        method: 降维方法，可选 'pca' 或 'tsne'
        title: 图标题
        figsize: 图像尺寸
    
    Returns:
        matplotlib Figure
    """
    # 展平特征: (B, 8, 16, 16, 16) -> (B*16*16*16, 8)
    B, C, D, H, W = shape.shape
    features_flat = shape.transpose(0, 2, 3, 4, 1).reshape(-1, C)  # (B*D*H*W, 8)
    
    # 采样以加速（如果点太多）
    if len(features_flat) > 10000:
        idx = np.random.choice(len(features_flat), 10000, replace=False)
        features_sample = features_flat[idx]
    else:
        features_sample = features_flat
    
    # 降维
    if method == "pca":
        reducer = PCA(n_components=3)
        features_3d = reducer.fit_transform(features_sample)
        explained_var = reducer.explained_variance_ratio_
        method_name = "PCA"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, random_state=42)
        features_3d = reducer.fit_transform(features_sample)
        explained_var = None
        method_name = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 创建可视化
    fig = plt.figure(figsize=figsize)
    
    # 3D 散点图
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(
        features_3d[:, 0], 
        features_3d[:, 1], 
        features_3d[:, 2],
        c=features_3d[:, 2],  # 用第三维作为颜色
        cmap='viridis',
        alpha=0.6,
        s=1
    )
    ax1.set_title(f'{title}\n({method_name} projection)')
    ax1.set_xlabel(f'{method_name} 1')
    ax1.set_ylabel(f'{method_name} 2')
    ax1.set_zlabel(f'{method_name} 3')
    plt.colorbar(scatter, ax=ax1, shrink=0.6)
    
    # 特征分布直方图
    ax2 = fig.add_subplot(122)
    ax2.hist(features_sample, bins=50, alpha=0.7, label=[f'Ch {i}' for i in range(C)])
    ax2.set_title('Feature Distribution per Channel')
    ax2.set_xlabel('Feature Value')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    
    # 打印解释方差（仅 PCA）
    if method == "pca" and explained_var is not None:
        print(f"PCA Explained Variance Ratios: {explained_var}")
        print(f"Total Explained Variance: {explained_var.sum():.2%}")
    
    return fig


def compare_sparse_coords(
    voxel_list: List[np.ndarray],
    labels: List[str],
    colors: List[str] = None,
    figsize: tuple = (16, 12),
):
    """
    对比多个稀疏点云。
    
    Args:
        voxel_list: 多个 (N, 3) 坐标数组列表
        labels: 每个点云的标签
        colors: 每个点云的颜色
        figsize: 图像尺寸
    
    Returns:
        matplotlib Figure
    """
    n = len(voxel_list)
    if colors is None:
        import seaborn as sns
        colors = sns.color_palette("husl", n)
    
    # 计算子图布局
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    
    fig = plt.figure(figsize=figsize)
    
    for i, (voxel, label, color) in enumerate(zip(voxel_list, labels, colors)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        visualize_sparse_coords(voxel, title=label, color=color, ax=ax)
    
    plt.tight_layout()
    return fig
