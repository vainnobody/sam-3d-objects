#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for the full 4-scene LERF-OVS Semantic Gaussians benchmark.
# It assumes you run from semantic-gaussians/ or from repo root via bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SG_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCENES=(figurines ramen teatime waldo_kitchen)

DATA_ROOT="${DATA_ROOT:-${SG_DIR}/../data/lerf_ovs}"
LABEL_ROOT="${LABEL_ROOT:-${DATA_ROOT}/label}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SG_DIR}/output/lerf_ovs}"
FUSION_ROOT="${FUSION_ROOT:-${SG_DIR}/fusion_lerf_ovs}"
EVAL_ROOT="${EVAL_ROOT:-${SG_DIR}/eval_result_lerf_ovs}"
MODEL_2D="${MODEL_2D:-openseg}"
TRAIN_ITERS="${TRAIN_ITERS:-30000}"
MASK_THRESH="${MASK_THRESH:-0.4}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_FUSION="${RUN_FUSION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_FUSION_ARGS="${EXTRA_FUSION_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

usage() {
  cat <<EOF
Usage:
  bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh

Environment overrides:
  DATA_ROOT        LERF-OVS scene root (default: ../data/lerf_ovs)
  LABEL_ROOT       LERF-OVS label root (default: \$DATA_ROOT/label)
  OUTPUT_ROOT      Trained 3DGS root (default: ./output/lerf_ovs)
  FUSION_ROOT      Fusion output root (default: ./fusion_lerf_ovs)
  EVAL_ROOT        Evaluation output root (default: ./eval_result_lerf_ovs)
  MODEL_2D         openseg or lseg (default: openseg)
  TRAIN_ITERS      3DGS training iterations (default: 30000)
  MASK_THRESH      Eval mask threshold (default: 0.4)
  RUN_TRAIN        1/0 whether to run train.py (default: 1)
  RUN_FUSION       1/0 whether to run fusion.py (default: 1)
  RUN_EVAL         1/0 whether to run eval_lerf_ovs.py (default: 1)
  EXTRA_TRAIN_ARGS Extra CLI overrides passed to train.py
  EXTRA_FUSION_ARGS Extra CLI overrides passed to fusion.py
  EXTRA_EVAL_ARGS  Extra CLI overrides passed to eval_lerf_ovs.py

Examples:
  RUN_TRAIN=0 RUN_FUSION=0 bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
  MODEL_2D=lseg TRAIN_ITERS=7000 bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_path() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing ${desc}: $path" >&2
    exit 1
  fi
}

require_path "$SG_DIR/train.py" "semantic-gaussians train.py"
require_path "$SG_DIR/fusion.py" "semantic-gaussians fusion.py"
require_path "$SG_DIR/eval_lerf_ovs.py" "semantic-gaussians eval_lerf_ovs.py"
require_path "$DATA_ROOT" "LERF-OVS data root"

mkdir -p "$OUTPUT_ROOT" "$FUSION_ROOT" "$EVAL_ROOT"

infer_img_dim() {
  local scene="$1"
  python3 - <<PY
from pathlib import Path
from PIL import Image
scene = Path(${DATA_ROOT@Q}) / ${scene@Q} / "images"
imgs = sorted([p for p in scene.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
if not imgs:
    raise SystemExit(f"No images found under {scene}")
width, height = Image.open(imgs[0]).size
print(f"[{width},{height}]")
PY
}

cd "$SG_DIR"

echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] LABEL_ROOT=$LABEL_ROOT"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[INFO] FUSION_ROOT=$FUSION_ROOT"
echo "[INFO] EVAL_ROOT=$EVAL_ROOT"
echo "[INFO] MODEL_2D=$MODEL_2D"

for scene in "${SCENES[@]}"; do
  SCENE_PATH="$DATA_ROOT/$scene"
  SCENE_OUTPUT="$OUTPUT_ROOT/$scene"
  SCENE_FUSION="$FUSION_ROOT/$scene"
  SCENE_LABEL="$LABEL_ROOT/$scene"

  require_path "$SCENE_PATH" "scene data for $scene"
  require_path "$SCENE_PATH/images" "images directory for $scene"
  require_path "$SCENE_PATH/sparse" "sparse COLMAP directory for $scene"

  IMG_DIM="$(infer_img_dim "$scene")"
  echo "[INFO] ===== Scene: $scene | fusion.img_dim=$IMG_DIM ====="

  if [[ "$RUN_TRAIN" == "1" ]]; then
    echo "[INFO] Training 3DGS for $scene"
    python train.py \
      scene.scene_path="$SCENE_PATH" \
      scene.colmap_images=images \
      scene.test_cameras=False \
      train.exp_name="lerf_ovs/$scene" \
      train.iterations="$TRAIN_ITERS" \
      ${EXTRA_TRAIN_ARGS}
  else
    echo "[INFO] Skipping train.py for $scene"
  fi

  if [[ "$RUN_FUSION" == "1" ]]; then
    require_path "$SCENE_OUTPUT/point_cloud" "trained point cloud directory for $scene"
    echo "[INFO] Running fusion.py for $scene"
    python fusion.py \
      scene.scene_path="$SCENE_PATH" \
      scene.colmap_images=images \
      scene.test_cameras=False \
      model.model_dir="$SCENE_OUTPUT" \
      fusion.out_dir="$SCENE_FUSION" \
      fusion.model_2d="$MODEL_2D" \
      fusion.img_dim="$IMG_DIM" \
      ${EXTRA_FUSION_ARGS}
  else
    echo "[INFO] Skipping fusion.py for $scene"
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    require_path "$SCENE_LABEL" "label directory for $scene"
    require_path "$SCENE_FUSION/0.pt" "fusion output for $scene"
    echo "[INFO] Running eval_lerf_ovs.py for $scene"
    python eval_lerf_ovs.py \
      eval.scene_names="[$scene]" \
      scene.scene_path="$DATA_ROOT" \
      model.model_dir="$OUTPUT_ROOT" \
      fusion.out_dir="$FUSION_ROOT" \
      eval.label_root="$LABEL_ROOT" \
      eval.output_dir="$EVAL_ROOT" \
      eval.model_2d="$MODEL_2D" \
      eval.mask_thresh="$MASK_THRESH" \
      ${EXTRA_EVAL_ARGS}
  else
    echo "[INFO] Skipping eval_lerf_ovs.py for $scene"
  fi

done

echo "[INFO] Done. Summary (if eval ran): $EVAL_ROOT/summary.json"
