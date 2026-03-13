#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SG_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WEIGHT_PATH="${WEIGHT_PATH:-${SG_DIR}/weights/openseg_exported_clip}"
RUN_SERVING_TEST=0
TEST_IMAGE=""

usage() {
  cat <<EOF
Usage:
  bash semantic-gaussians/tools/diagnose_openseg_env.sh [--weight-path PATH] [--run-serving-test --image PATH]

This script diagnoses TensorFlow/OpenSeg GPU runtime issues such as:
  - UnimplementedError: Graph execution error
  - DNN library is not found
  - OpenSeg SavedModel / cuDNN / CUDA runtime mismatch

Options:
  --weight-path PATH   Override OpenSeg SavedModel path (default: ./weights/openseg_exported_clip relative to semantic-gaussians/)
  --run-serving-test   Run a minimal TensorFlow SavedModel inference test
  --image PATH         Test image for --run-serving-test

Environment:
  WEIGHT_PATH          Alternative way to override the SavedModel path
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weight-path)
      WEIGHT_PATH="$2"
      shift 2
      ;;
    --run-serving-test)
      RUN_SERVING_TEST=1
      shift
      ;;
    --image)
      TEST_IMAGE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[FAIL] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$RUN_SERVING_TEST" == "1" && -z "$TEST_IMAGE" ]]; then
  echo "[FAIL] --run-serving-test requires --image PATH" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
STATUS_OK=0
STATUS_WARN=0
STATUS_FAIL=0

ok() { echo "[OK] $*"; STATUS_OK=$((STATUS_OK + 1)); }
warn() { echo "[WARN] $*"; STATUS_WARN=$((STATUS_WARN + 1)); }
fail() { echo "[FAIL] $*"; STATUS_FAIL=$((STATUS_FAIL + 1)); }
section() { printf '\n===== %s =====\n' "$*"; }

cd "$SG_DIR"

section "Basic Environment"
echo "Working directory: $SG_DIR"
echo "Python binary: $(command -v "$PYTHON_BIN" || echo "$PYTHON_BIN (not found in PATH)")"
echo "Weight path: $WEIGHT_PATH"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<empty>}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<empty>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi found"
  nvidia-smi || warn "nvidia-smi exists but failed to query GPU state"
else
  fail "nvidia-smi not found; NVIDIA driver/GPU may be unavailable"
fi

if command -v nvcc >/dev/null 2>&1; then
  ok "nvcc found"
  nvcc --version | sed -n '1,5p'
else
  warn "nvcc not found; CUDA toolkit may be absent from PATH"
fi

section "OpenSeg SavedModel Check"
if [[ -d "$WEIGHT_PATH" ]]; then
  ok "OpenSeg SavedModel directory exists"
  [[ -f "$WEIGHT_PATH/saved_model.pb" ]] && ok "saved_model.pb found" || warn "saved_model.pb not found under $WEIGHT_PATH"
  [[ -d "$WEIGHT_PATH/variables" ]] && ok "variables/ directory found" || warn "variables/ directory not found under $WEIGHT_PATH"
else
  fail "Missing OpenSeg SavedModel directory: $WEIGHT_PATH"
fi

section "CUDA/cuDNN Library Check"
LIB_HITS="$("$PYTHON_BIN" - <<'PY'
import ctypes.util
libs = ["cudnn", "cublas", "cudart"]
for lib in libs:
    print(f"{lib}:{ctypes.util.find_library(lib) or ''}")
PY
)"
while IFS= read -r line; do
  name="${line%%:*}"
  hit="${line#*:}"
  if [[ -n "$hit" ]]; then
    ok "Library lookup found $name -> $hit"
  else
    warn "Library lookup did not find $name in dynamic linker search path"
  fi
done <<< "$LIB_HITS"

section "TensorFlow Runtime Check"
TF_REPORT_FILE="$(mktemp)"
set +e
"$PYTHON_BIN" - "$WEIGHT_PATH" >"$TF_REPORT_FILE" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

weight_path = Path(sys.argv[1])

report = {
    "tf_import_ok": False,
    "tf_version": None,
    "tf_build_info": {},
    "gpus": [],
    "saved_model_load_ok": False,
    "saved_model_error": None,
}

try:
    import tensorflow as tf
    report["tf_import_ok"] = True
    report["tf_version"] = tf.__version__
    try:
        report["tf_build_info"] = tf.sysconfig.get_build_info()
    except Exception as exc:
        report["tf_build_info"] = {"error": repr(exc)}
    try:
        report["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception as exc:
        report["gpus_error"] = repr(exc)
    if weight_path.is_dir():
        try:
            tf.saved_model.load(str(weight_path))
            report["saved_model_load_ok"] = True
        except Exception as exc:
            report["saved_model_error"] = repr(exc)
except Exception as exc:
    report["tf_import_error"] = repr(exc)

print(json.dumps(report))
PY
TF_EXIT=$?
set -e

TF_JSON="$(tail -n 1 "$TF_REPORT_FILE" || true)"
if [[ $TF_EXIT -ne 0 || -z "$TF_JSON" ]]; then
  fail "TensorFlow runtime probe crashed"
  sed -n '1,200p' "$TF_REPORT_FILE"
else
  "$PYTHON_BIN" - "$TF_JSON" <<'PY'
import json, sys
report = json.loads(sys.argv[1])
print(json.dumps(report, indent=2, ensure_ascii=False))
PY
  TF_IMPORT_OK="$("$PYTHON_BIN" - "$TF_JSON" <<'PY'
import json, sys
print("1" if json.loads(sys.argv[1]).get("tf_import_ok") else "0")
PY
)"
  TF_GPU_COUNT="$("$PYTHON_BIN" - "$TF_JSON" <<'PY'
import json, sys
print(len(json.loads(sys.argv[1]).get("gpus", [])))
PY
)"
  TF_SAVEDMODEL_OK="$("$PYTHON_BIN" - "$TF_JSON" <<'PY'
import json, sys
print("1" if json.loads(sys.argv[1]).get("saved_model_load_ok") else "0")
PY
)"
  TF_VERSION="$("$PYTHON_BIN" - "$TF_JSON" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("tf_version") or "")
PY
)"

  if [[ "$TF_IMPORT_OK" == "1" ]]; then
    ok "TensorFlow import succeeded (version: ${TF_VERSION:-unknown})"
  else
    fail "TensorFlow import failed"
  fi
  if [[ "$TF_GPU_COUNT" -gt 0 ]]; then
    ok "TensorFlow sees $TF_GPU_COUNT GPU device(s)"
  else
    fail "TensorFlow does not see any GPU devices"
  fi
  if [[ "$TF_SAVEDMODEL_OK" == "1" ]]; then
    ok "tf.saved_model.load succeeded for OpenSeg"
  else
    warn "tf.saved_model.load did not succeed; see JSON above for error details"
  fi
fi

rm -f "$TF_REPORT_FILE"

if [[ "$RUN_SERVING_TEST" == "1" ]]; then
  section "OpenSeg serving_default Probe"
  SERVE_REPORT_FILE="$(mktemp)"
  set +e
  "$PYTHON_BIN" - "$WEIGHT_PATH" "$TEST_IMAGE" >"$SERVE_REPORT_FILE" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

weight_path = Path(sys.argv[1])
image_path = Path(sys.argv[2])
report = {"serving_test_ok": False}

try:
    import tensorflow as tf2
    import tensorflow.compat.v1 as tf
    from tensorflow import io
    if not image_path.is_file():
      raise FileNotFoundError(f"Test image not found: {image_path}")
    model = tf2.saved_model.load(str(weight_path), tags=[tf.saved_model.tag_constants.SERVING])
    with io.gfile.GFile(str(image_path), "rb") as f:
        file_bytes = f.read()
    text_emb = tf.zeros([1, 1, 768])
    outputs = model.signatures["serving_default"](
        inp_image_bytes=tf.convert_to_tensor(file_bytes),
        inp_text_emb=text_emb,
    )
    report["serving_test_ok"] = True
    report["keys"] = sorted(outputs.keys())
except Exception as exc:
    report["serving_test_error"] = repr(exc)
print(json.dumps(report))
PY
  SERVE_EXIT=$?
  set -e
  SERVE_JSON="$(tail -n 1 "$SERVE_REPORT_FILE" || true)"
  if [[ $SERVE_EXIT -ne 0 || -z "$SERVE_JSON" ]]; then
    fail "serving_default probe crashed"
    sed -n '1,220p' "$SERVE_REPORT_FILE"
  else
    "$PYTHON_BIN" - "$SERVE_JSON" <<'PY'
import json, sys
print(json.dumps(json.loads(sys.argv[1]), indent=2, ensure_ascii=False))
PY
    SERVE_OK="$("$PYTHON_BIN" - "$SERVE_JSON" <<'PY'
import json, sys
print("1" if json.loads(sys.argv[1]).get("serving_test_ok") else "0")
PY
)"
    if [[ "$SERVE_OK" == "1" ]]; then
      ok "serving_default probe succeeded"
    else
      fail "serving_default probe failed"
    fi
  fi
  rm -f "$SERVE_REPORT_FILE"
fi

section "Diagnosis Summary"
echo "OK checks:   $STATUS_OK"
echo "WARN checks: $STATUS_WARN"
echo "FAIL checks: $STATUS_FAIL"

echo
echo "Recommended fixes:"
if [[ ! -d "$WEIGHT_PATH" ]]; then
  echo "  1. Download the OpenSeg exported SavedModel into: $WEIGHT_PATH"
fi
echo "  2. Use the semantic-gaussians environment expected by this repo:"
echo "     - semantic-gaussians/environment.yml"
echo "     - semantic-gaussians/requirements.txt (tensorflow[and-cuda]==2.14.0)"
echo "  3. If TensorFlow sees no GPU or serving_default reports 'DNN library is not found':"
echo "     - verify NVIDIA driver with: nvidia-smi"
echo "     - verify CUDA/cuDNN runtime libraries are on LD_LIBRARY_PATH"
echo "     - ensure TensorFlow, CUDA, and cuDNN versions are mutually compatible"
echo "  4. If you need to continue the benchmark before fixing OpenSeg, use:"
echo "     MODEL_2D=lseg bash semantic-gaussians/tools/run_lerf_ovs_eval_all.sh"

if [[ $STATUS_FAIL -gt 0 ]]; then
  exit 1
fi
