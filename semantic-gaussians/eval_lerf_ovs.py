import json
import logging
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from model import GaussianModel, render_chn
from scene import Scene
from utils.system_utils import searchForMaxIteration, set_seed


def build_text_model(model_name: str):
    model_name = model_name.lower().replace("_", "")
    if model_name == "lseg":
        from model.lseg_predictor import LSeg

        return LSeg(None)
    if model_name == "openseg":
        from model.openseg_predictor import OpenSeg

        return OpenSeg(None, "ViT-L/14@336px")
    raise ValueError(f"Unsupported eval.model_2d: {model_name}. Expected one of: openseg, lseg")


def parse_scene_names(config) -> list[str]:
    configured = config.eval.get("scene_names")
    if configured:
        return [str(name) for name in configured]

    root = Path(config.model.model_dir)
    if not root.exists():
        raise FileNotFoundError(f"model.model_dir does not exist: {root}")
    scene_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not scene_names:
        raise RuntimeError(f"No scene directories found under {root}")
    return scene_names


def load_gt_scene(scene_label_dir: Path) -> dict[str, dict[str, Any]]:
    if not scene_label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing LERF-OVS label directory: {scene_label_dir}. "
            "You need lerf_ovs/label/<scene>/frame_*.json annotations for benchmark evaluation."
        )

    frame_data: dict[str, dict[str, Any]] = {}
    for json_path in sorted(scene_label_dir.glob("frame_*.json")):
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        image_name = payload.get("info", {}).get("name")
        if not image_name:
            raise RuntimeError(f"Invalid GT json without info.name: {json_path}")
        frame_stem = Path(image_name).stem
        height = int(payload["info"]["height"])
        width = int(payload["info"]["width"])

        objects = defaultdict(dict)
        for obj in payload.get("objects", []):
            label = str(obj["category"])
            bbox = np.asarray(obj["bbox"], dtype=np.float32).reshape(-1, 4)
            polygon = np.asarray(obj["segmentation"], dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            if "mask" in objects[label]:
                objects[label]["mask"] = np.logical_or(objects[label]["mask"], mask).astype(np.uint8)
                objects[label]["bboxes"] = np.concatenate([objects[label]["bboxes"], bbox], axis=0)
            else:
                objects[label]["mask"] = mask
                objects[label]["bboxes"] = bbox

        image_path = scene_label_dir / image_name
        frame_data[frame_stem] = {
            "json_path": json_path,
            "image_path": image_path,
            "height": height,
            "width": width,
            "objects": dict(objects),
        }

    if not frame_data:
        raise RuntimeError(f"No frame_*.json annotations found under {scene_label_dir}")
    return frame_data


def load_scene_bundle(config, scene_name: str, text_model):
    scene_config = deepcopy(config)
    scene_config.scene.scene_path = os.path.join(scene_config.scene.scene_path, scene_name)
    scene_config.model.model_dir = os.path.join(scene_config.model.model_dir, scene_name)
    scene_config.fusion.out_dir = os.path.join(scene_config.fusion.out_dir, scene_name)

    scene = Scene(scene_config.scene)
    gaussians = GaussianModel(scene_config.model.sh_degree)
    loaded_iter = scene_config.model.load_iteration
    if loaded_iter == -1:
        loaded_iter = searchForMaxIteration(os.path.join(scene_config.model.model_dir, "point_cloud"))
    gaussians.load_ply(
        os.path.join(
            scene_config.model.model_dir,
            "point_cloud",
            f"iteration_{loaded_iter}",
            "point_cloud.ply",
        )
    )
    gaussians.create_semantic(text_model.embedding_dim)

    feature_path = Path(scene_config.fusion.out_dir) / "0.pt"
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Missing fusion feature for scene '{scene_name}': {feature_path}. "
            "Run fusion.py first."
        )
    fusion = torch.load(feature_path, map_location="cpu")
    feat = fusion["feat"].float().cuda()
    mask_full = fusion["mask_full"].bool().cuda()
    gaussians._features_semantic.zero_()
    gaussians._features_semantic[mask_full] = feat

    bg_color = [1] * text_model.embedding_dim if scene_config.scene.white_background else [0] * text_model.embedding_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras()
    view_map = {view.image_name: view for view in views}
    return scene_config, gaussians, background, view_map


def normalize_map(relevancy: np.ndarray) -> np.ndarray:
    min_value = float(relevancy.min())
    max_value = float(relevancy.max())
    denom = max(max_value - min_value, 1e-8)
    return (relevancy - min_value) / denom


def smooth_relevancy(relevancy: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / float(kernel_size * kernel_size)
    avg_filtered = cv2.filter2D(relevancy.astype(np.float32), -1, kernel)
    return 0.5 * (avg_filtered + relevancy.astype(np.float32))


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), (mask.astype(np.uint8) * 255))


def save_heatmap_overlay(base_rgb: np.ndarray, relevancy: np.ndarray, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    heat_uint8 = np.clip(relevancy * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (0.65 * colored + 0.35 * base_rgb).clip(0, 255).astype(np.uint8)
    imageio.imwrite(output_prefix.with_suffix(".png"), colored)
    imageio.imwrite(output_prefix.parent.parent / "composited" / output_prefix.name.replace("heatmap", "composited") , overlay)


def draw_localization(base_rgb: np.ndarray, peak_xy: tuple[int, int], bboxes: np.ndarray, output_path: Path) -> None:
    canvas = cv2.cvtColor(base_rgb.copy(), cv2.COLOR_RGB2BGR)
    px, py = peak_xy
    for box in bboxes.reshape(-1, 4).astype(int):
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (px, py), 8, (30, 30, 220), -1, lineType=cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def render_prompt_maps(view, gaussians, scene_config, background, text_features: torch.Tensor, render_hw: tuple[int, int]) -> np.ndarray:
    width, height = render_hw
    view.cuda()
    rendering = render_chn(
        view,
        gaussians,
        scene_config.pipeline,
        background,
        num_channels=text_features.shape[1],
        override_color=gaussians._features_semantic,
        override_shape=[width, height],
    )["render"]
    rendering = rendering / (rendering.norm(dim=0, keepdim=True) + 1e-8)
    sim = torch.einsum("cq,qhw->chw", text_features, rendering)
    return sim.detach().cpu().numpy()


def evaluate_label(prompt_index: int, prompt: str, sim_maps: np.ndarray, gt_mask: np.ndarray, gt_boxes: np.ndarray, base_rgb: np.ndarray, frame_output_dir: Path, mask_thresh: float, smoothing_kernel: int) -> dict[str, Any]:
    prompt_sim = sim_maps[prompt_index]
    normalized = normalize_map(prompt_sim)
    smoothed = smooth_relevancy(normalized, smoothing_kernel)
    peak_index = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    peak_xy = (int(peak_index[1]), int(peak_index[0]))

    pred_mask = (smoothed > mask_thresh).astype(np.uint8)
    intersection = float(np.logical_and(gt_mask > 0, pred_mask > 0).sum())
    union = float(np.logical_or(gt_mask > 0, pred_mask > 0).sum())
    iou = intersection / max(union, 1.0)

    localization_hit = False
    for box in gt_boxes.reshape(-1, 4):
        x1, y1, x2, y2 = box.tolist()
        if x1 <= peak_xy[0] <= x2 and y1 <= peak_xy[1] <= y2:
            localization_hit = True
            break

    heatmap_path = frame_output_dir / "heatmap" / f"{prompt}.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    heat_uint8 = np.clip(smoothed * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (0.65 * colored_rgb + 0.35 * base_rgb).clip(0, 255).astype(np.uint8)
    imageio.imwrite(heatmap_path, colored_rgb)
    composited_path = frame_output_dir / "composited" / f"{prompt}.png"
    composited_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(composited_path, overlay)
    save_mask(gt_mask, frame_output_dir / "gt" / f"{prompt}.png")
    save_mask(pred_mask, frame_output_dir / f"chosen_{prompt}.png")
    draw_localization(base_rgb, peak_xy, gt_boxes, frame_output_dir / "localization" / f"{prompt}.png")

    return {
        "prompt": prompt,
        "iou": iou,
        "localization_hit": localization_hit,
        "peak_xy": [peak_xy[0], peak_xy[1]],
        "bbox_count": int(gt_boxes.reshape(-1, 4).shape[0]),
        "mask_pixels_gt": int((gt_mask > 0).sum()),
        "mask_pixels_pred": int((pred_mask > 0).sum()),
    }


def evaluate_scene(config, scene_name: str, text_model) -> dict[str, Any]:
    logger = logging.getLogger("eval_lerf_ovs")
    label_root = Path(config.eval.label_root) / scene_name
    gt_frames = load_gt_scene(label_root)
    scene_config, gaussians, background, view_map = load_scene_bundle(config, scene_name, text_model)

    missing_frames = sorted(set(gt_frames) - set(view_map))
    if missing_frames:
        raise RuntimeError(
            f"Scene '{scene_name}' has GT frames not found in loaded cameras: {missing_frames[:10]}"
            + (" ..." if len(missing_frames) > 10 else "")
        )

    output_root = Path(config.eval.output_dir) / scene_name
    output_root.mkdir(parents=True, exist_ok=True)

    scene_results = []
    total_hits = 0
    total_prompts = 0
    all_ious = []

    for frame_name in tqdm(sorted(gt_frames.keys()), desc=f"eval {scene_name}"):
        frame_gt = gt_frames[frame_name]
        view = view_map[frame_name]
        prompt_names = sorted(frame_gt["objects"].keys())
        text_features = text_model.extract_text_feature(["other"] + prompt_names).float().cuda()
        sim_maps = render_prompt_maps(
            view,
            gaussians,
            scene_config,
            background,
            text_features,
            render_hw=(int(frame_gt["width"]), int(frame_gt["height"])),
        )[1:]

        if frame_gt["image_path"].is_file():
            base_rgb = cv2.cvtColor(cv2.imread(str(frame_gt["image_path"])), cv2.COLOR_BGR2RGB)
        else:
            base_rgb = cv2.cvtColor(cv2.imread(str(view.image_path)), cv2.COLOR_BGR2RGB)
            if base_rgb.shape[0] != frame_gt["height"] or base_rgb.shape[1] != frame_gt["width"]:
                base_rgb = cv2.resize(base_rgb, (int(frame_gt["width"]), int(frame_gt["height"])), interpolation=cv2.INTER_LINEAR)

        frame_output_dir = output_root / frame_name
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        frame_records = []
        for prompt_index, prompt in enumerate(prompt_names):
            record = evaluate_label(
                prompt_index=prompt_index,
                prompt=prompt,
                sim_maps=sim_maps,
                gt_mask=frame_gt["objects"][prompt]["mask"],
                gt_boxes=frame_gt["objects"][prompt]["bboxes"],
                base_rgb=base_rgb,
                frame_output_dir=frame_output_dir,
                mask_thresh=float(config.eval.mask_thresh),
                smoothing_kernel=int(config.eval.smoothing_kernel),
            )
            frame_records.append(record)
            all_ious.append(record["iou"])
            total_prompts += 1
            total_hits += int(record["localization_hit"])

        scene_results.append({"frame": frame_name, "prompts": frame_records})

    metrics = {
        "scene": scene_name,
        "num_frames": len(scene_results),
        "num_prompts": total_prompts,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "localization_accuracy": float(total_hits / total_prompts) if total_prompts else 0.0,
        "frames": scene_results,
    }
    with open(output_root / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with open(output_root / "metrics.txt", "w", encoding="utf-8") as handle:
        handle.write(f"scene: {scene_name}\n")
        handle.write(f"num_frames: {metrics['num_frames']}\n")
        handle.write(f"num_prompts: {metrics['num_prompts']}\n")
        handle.write(f"mean_iou: {metrics['mean_iou']:.4f}\n")
        handle.write(f"localization_accuracy: {metrics['localization_accuracy']:.4f}\n")
    logger.info(
        "Scene %s | frames=%d prompts=%d mean_iou=%.4f localization=%.4f",
        scene_name,
        metrics["num_frames"],
        metrics["num_prompts"],
        metrics["mean_iou"],
        metrics["localization_accuracy"],
    )
    return metrics


def main():
    config = OmegaConf.load("./config/eval_lerf_ovs.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("eval_lerf_ovs")
    logger.info("\n%s", OmegaConf.to_yaml(config))

    if config.eval.source != "fusion":
        raise ValueError("Only eval.source=fusion is supported in the first LERF-OVS SG benchmark implementation")

    set_seed(config.pipeline.seed)
    text_model = build_text_model(config.eval.model_2d)
    scene_names = parse_scene_names(config)
    summary = {"scenes": [], "overall": {}}

    for scene_name in scene_names:
        metrics = evaluate_scene(config, scene_name, text_model)
        summary["scenes"].append({
            "scene": scene_name,
            "mean_iou": metrics["mean_iou"],
            "localization_accuracy": metrics["localization_accuracy"],
            "num_frames": metrics["num_frames"],
            "num_prompts": metrics["num_prompts"],
        })

    if summary["scenes"]:
        summary["overall"] = {
            "scene_count": len(summary["scenes"]),
            "mean_iou": float(np.mean([item["mean_iou"] for item in summary["scenes"]])),
            "localization_accuracy": float(np.mean([item["localization_accuracy"] for item in summary["scenes"]])),
            "num_frames": int(sum(item["num_frames"] for item in summary["scenes"])),
            "num_prompts": int(sum(item["num_prompts"] for item in summary["scenes"])),
        }

    output_root = Path(config.eval.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(output_root / "summary.txt", "w", encoding="utf-8") as handle:
        for item in summary["scenes"]:
            handle.write(
                f"{item['scene']}: mean_iou={item['mean_iou']:.4f}, localization_accuracy={item['localization_accuracy']:.4f}, "
                f"frames={item['num_frames']}, prompts={item['num_prompts']}\n"
            )
        if summary.get("overall"):
            overall = summary["overall"]
            handle.write(
                f"overall: mean_iou={overall['mean_iou']:.4f}, localization_accuracy={overall['localization_accuracy']:.4f}, "
                f"scene_count={overall['scene_count']}, frames={overall['num_frames']}, prompts={overall['num_prompts']}\n"
            )
    logger.info("Saved summary to %s", output_root / "summary.json")


if __name__ == "__main__":
    main()
