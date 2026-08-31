#!/usr/bin/env bash
#
# migrate_to_src_layout.sh
#
# One-shot refactoring script that migrates the legacy flat directory layout
# into the clean `src/`-based architecture.
#
# Safe to run on a fresh clone of the ORIGINAL layout (before any manual
# migration). It is idempotent: directories already migrated are skipped,
# and only empty legacy folders are removed.
#
# Usage:
#   bash scripts/migrate_to_src_layout.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf '\033[1;34m[refactor]\033[0m %s\n' "$*"; }

# ---------- 1. Create target directories ----------
say "Creating target directory structure..."
mkdir -p \
  src/training \
  src/inference \
  src/evaluation \
  src/data \
  src/export \
  src/deployment/rknn \
  models/rknn \
  models/onnx \
  models/tflite \
  config \
  docs \
  tests \
  scripts \
  data \
  outputs \
  requirements

touch src/__init__.py \
  src/training/__init__.py \
  src/inference/__init__.py \
  src/evaluation/__init__.py \
  src/data/__init__.py \
  src/export/__init__.py \
  src/deployment/__init__.py \
  src/deployment/rknn/__init__.py \
  tests/__init__.py \
  models/.gitkeep models/rknn/.gitkeep models/onnx/.gitkeep models/tflite/.gitkeep \
  data/.gitkeep outputs/.gitkeep config/.gitkeep scripts/.gitkeep tests/.gitkeep

# ---------- 2. File migration map ----------
# old_path -> new_path  (handled by `mv`; git mv is used if tracked)
_mv() { # _mv <src> <dst>
  local src="$1" dst="$2"
  [ -e "$src" ] || { say "skip (not found): $src"; return 0; }
  [ -e "$dst" ] && { say "skip (exists):   $dst"; return 0; }
  if git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
    mkdir -p "$(dirname "$dst")"
    git mv "$src" "$dst"
  else
    mv "$src" "$dst"
  fi
  say "moved:            $src -> $dst"
}

say "Migrating Python scripts into src/ ..."
_mv TrainingAndTests/model_train.py           src/training/model_train.py
_mv TrainingAndTests/model_test.py            src/inference/model_test.py
_mv TrainingAndTests/opt_test_model.py        src/inference/opt_test_model.py
_mv TrainingAndTests/benchmark_yolo_video.py  src/inference/benchmark_yolo_video.py
_mv TrainingAndTests/fpn-detection.py         src/inference/fpn_detection.py
_mv TrainingAndTests/test_several_models.py   src/evaluation/test_several_models.py
_mv TrainingAndTests/Metric_graph.py          src/evaluation/metric_graph.py
_mv TrainingAndTests/cudaTest.py              src/evaluation/cuda_test.py
_mv augmentation/augmentation.py              src/data/augmentation.py
_mv augmentation/cat_vid.py                   src/data/cat_vid.py
_mv augmentation/resize_and_crop.py           src/data/resize_and_crop.py
_mv augmentation/resize_and_pad.py            src/data/resize_and_pad.py
_mv augmentation/resize_images.py             src/data/resize_images.py
_mv ConvertToFormat/ConvertToFormat_rknn.py   src/export/convert_to_format_rknn.py
_mv ConvertToFormat/ConvertToFormat_TF_lite.py src/export/convert_to_format_tflite.py
_mv RKNN/test_rknn_model_Image.py             src/deployment/rknn/test_rknn_model_image.py
_mv RKNN/test_rknn_model_Video.py             src/deployment/rknn/test_rknn_model_video.py

say "Consolidating model artifacts into models/ ..."
# Preserve copies of untracked artifacts (they are gitignored; do NOT git rm them
# here - removal from the index is a separate, explicit step).
mkdir -p models/tflite/best_yolo11n_tflite_model models/rknn/best_yolo11n_rknn_model
_mv ALL_MODEL/best_yolo11n.pt                 models/best_yolo11n.pt
_mv ALL_MODEL/best_yolo11n.onnx               models/onnx/best_yolo11n.onnx
_mv ALL_MODEL/best_yolo11n.engine             models/onnx/best_yolo11n.engine
_mv ALL_MODEL/best_yolo11n_TFlite_model       models/tflite/best_yolo11n_tflite_model
_mv RKNN/best_yolo11n_rknn_model              models/rknn/best_yolo11n_rknn_model

# ---------- 3. Remove now-empty legacy folders (safely) ----------
say "Removing empty legacy folders..."
for d in ALL_MODEL ConvertToFormat RKNN TrainingAndTests augmentation; do
  if [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
    rmdir "$d" && say "removed empty dir: $d"
  fi
done

say "Migration complete."
echo
say "Next steps:"
echo "    1. Review the updated README.md and ARCHITECTURE.md"
echo "    2. If model binaries were previously tracked, remove them from git:"
echo "         git rm -r --cached models/   # keeps files on disk"
echo "    3. Inspect hardcoded source paths (scripts moved, so '../..' depth changed)."
