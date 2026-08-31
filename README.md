# YOLO Multi-Platform Inference

A comprehensive **YOLO object-detection pipeline** for training, evaluating, testing, optimizing, and deploying tank-detection models across multiple platforms: **PyTorch / CUDA**, **TensorFlow Lite**, **TensorRT (ONNX/Engine)**, and **RKNN** (Rockchip NPU: RK3588 et al.).

The repository follows a **clean, `src/`-based, domain-oriented layout** so that source code, model artifacts, configuration, tests, documentation, and runtime outputs are logically isolated.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Key Features](#key-features)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Environment Variables](#environment-variables)
6. [Common Workflows](#common-workflows)
7. [Model Export & Deployment](#model-export--deployment)
8. [Output Files & Logs](#output-files--logs)
9. [Contributing & Architecture](#contributing--architecture)
10. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
yolo-multi-platform-inference/
├── src/                          # All Python source code (domain-oriented)
│   ├── training/                 # Model training (model_train.py)
│   ├── inference/                # Inference & benchmarking (model_test, opt_test,
│   │                             #   benchmark_yolo_video, fpn_detection)
│   ├── evaluation/               # Model comparison, metrics, CUDA check
│   ├── data/                     # Augmentation + image/video preprocessing
│   ├── export/                   # Model export to RKNN / TensorFlow Lite
│   └── deployment/
│       └── rknn/                 # Runtime testing of RKNN-converted models
├── models/                       # Model artifacts (gitignored, dirs kept via .gitkeep)
│   ├── best_yolo11n.pt           #   PyTorch checkpoint
│   ├── onnx/                     #   .onnx / TensorRT .engine
│   ├── tflite/                   #   TensorFlow Lite (SavedModel + .tflite)
│   └── rknn/                     #   RKNN (.rknn + metadata)
├── config/                       # Configuration templates
│   └── data.yaml.template        #   Ultralytics dataset config template
├── scripts/                      # Ops / maintenance scripts (e.g. migration tool)
├── tests/                        # Unit / integration tests
├── docs/                         # Architecture & contribution guidelines
├── data/                         # Datasets & media (NOT versioned)
├── outputs/                      # Runtime results, logs, videos (NOT versioned)
├── requirements/                 # Split dependency groups
│   ├── base.txt
│   ├── rknn.txt
│   ├── tflite.txt
│   └── dev.txt
├── README.md                     # This file
├── ARCHITECTURE.md               # Contributor placement guidelines
└── ...
```

> **Convention:** source code lives under `src/`; nothing runnable is committed at the repository root. Model binaries, datasets, and generated outputs are **gitignored** — their directories are kept in the repo with `.gitkeep` files.

---

## Key Features

### 🎯 Core Capabilities

- **Multi-platform support** — train on PyTorch/CUDA, deploy to CPU, GPU, TensorFlow Lite, and RKNN (Rockchip NPU).
- **Advanced detection** — standard detection, FPN (Feature Pyramid Network) multi-scale detection, and image slicing for very large images.
- **Comprehensive testing** — FPS/memory benchmarking, per-frame CSV logging, confidence histograms, and warm-up handling.
- **Data augmentation** — mirror, safe, balanced, and `augment+` strategies.
- **Model comparison** — side-by-side mAP evaluation of several trained models.
- **Model export** — convert trained weights to TensorFlow Lite and RKNN formats.

### 📊 Monitoring & Metrics

- Real-time FPS tracking and RAM/VRAM monitoring
- Detection confidence analysis
- Frame-by-frame CSV logging
- Training-metric visualization (`metric_graph.py`)
- Performance graphs and histogram export

---

## Prerequisites

### Hardware

| Platform | Requirement |
| --- | --- |
| CPU | 8 GB RAM (minimum) |
| GPU | NVIDIA GPU with CUDA (16 GB+ RAM recommended) |
| RKNN | Rockchip RK3588 / RK3576 / RK3566 family board |

### Dependencies

```bash
# Core (training + inference)
pip install -r requirements/base.txt

# Optional: RKNN export/deploy
pip install -r requirements/rknn.txt

# Optional: TensorFlow Lite export/deploy
pip install -r requirements/tflite.txt

# Development tooling (linting, tests)
pip install -r requirements/dev.txt
```

---

## Quick Start

### 1. Set up the environment

```bash
git clone <your-repo-url>
cd yolo-multi-platform-inference
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements/base.txt
```

### 2. Configure your dataset

Copy the template and fill in the absolute paths:

```bash
cp config/data.yaml.template data/data.yaml
# edit data/data.yaml -> set `path`, `train/val/test`, and `names`
```

Place your trained PyTorch checkpoint in `models/` (e.g. `models/best_yolo11n.pt`).

### 3. Run a training cycle

```bash
python src/training/model_train.py
```

### 4. Run inference on a video

```bash
python src/inference/model_test.py \
    --source data/test_video.mp4 \
    --model models/best_yolo11n.pt \
    --conf 0.25 --iou 0.45 --device 0 \
    --save-results --memory-monitor
```

### 5. Export for a target platform

```bash
# RKNN (run on the RKNN toolchain host)
python src/export/convert_to_format_rknn.py

# TensorFlow Lite
python src/export/convert_to_format_tflite.py
```

See [Common Workflows](#common-workflows) for the full pipelines.

---

## Environment Variables

The library currently relies on **CLI arguments and in-script path constants** rather than environment variables. The following variables are used / recommended:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | Restrict which GPUs are visible to PyTorch | *(all)* |
| `DATASET_ROOT` *(optional)* | Override the dataset root referenced in `data.yaml` | *(from `data.yaml`)* |
| `MODELS_DIR` *(optional)* | Override the default `models/` directory | `models/` |
| `OUTPUTS_DIR` *(optional)* | Override the default output/results directory | `outputs/` |

> Because the scripts currently hardcode model/data paths, run them from the **repository root** so relative paths like `models/...` and `data/...` resolve correctly. Standardizing on environment variables is a planned enhancement (see `ARCHITECTURE.md` → "Roadmap").

---

## Common Workflows

### 1. Complete training pipeline

```bash
# 1. Prepare data
python src/data/cat_vid.py                       # extract frames from videos
python src/data/augmentation.py data/frames --type augmentplus
python src/data/resize_images.py data/frames --output_folder data/processed

# 2. Train model
python src/training/model_train.py               # edit hyperparameters in script

# 3. Visualize metrics
python src/evaluation/metric_graph.py             # plot results.csv

# 4. Compare several models
python src/evaluation/test_several_models.py
```

### 2. Testing & benchmarking pipeline

```bash
python src/inference/model_test.py --source data/test.mp4 --model models/best_yolo11n.pt --save-results
python src/inference/benchmark_yolo_video.py      # edit paths in script
python src/inference/opt_test_model.py            # pure-speed test, no visualization
python src/inference/fpn_detection.py --source data/test.mp4 --model models/best_yolo11n.pt \
    --use-fpn --fpn-scales 0.5,1.0,1.5 --save-results
```

### 3. Deployment pipeline

```bash
python src/export/convert_to_format_rknn.py       # 1. export to RKNN
python src/deployment/rknn/test_rknn_model_image.py   # 2. test on image
python src/deployment/rknn/test_rknn_model_video.py   # 3. test on video
python src/export/convert_to_format_tflite.py     # 4. export to TFLite
```

---

## Model Export & Deployment

| Format | Script | Targets |
| --- | --- | --- |
| RKNN | `src/export/convert_to_format_rknn.py` | rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118 |
| TensorFlow Lite | `src/export/convert_to_format_tflite.py` | mobile / embedded |

RKNN-converted models are saved under `models/rknn/`; TFLite outputs land under `models/tflite/`.

---

## Output Files & Logs

All runtime artifacts are written under the gitignored `outputs/` directory (or the working directory when a script defines its own `output_dir`).

- **CSV logs**: per-frame `Timestamp,Frame,ProcessingTime,InstantFPS,Detections,...`
- **Performance graphs**: `fps_plot.png`, `confidence_histogram.png`
- **Annotated results**: annotated video files and per-frame images
- **RKNN test outputs**: `*_detected_*.jpg`, `*_detections_*.json`, `*_summary_*.txt`

---

## Contributing & Architecture

Before adding new modules, scripts, tests, or utilities, read the **[ARCHITECTURE.md](./ARCHITECTURE.md)** file. It defines **where each type of code belongs** and the naming/quality conventions to follow.

---

## Troubleshooting

1. **CUDA out of memory** — reduce batch size, lower input resolution, or enable frame skipping (`--frame-skip`).
2. **Slow performance** — verify GPU utilization, enable `cudnn.benchmark`, or use the optimized test script.
3. **Poor detection quality** — adjust confidence threshold, enable FPN for small objects, or increase epochs/augmentation.
4. **Augmentation artifacts** — use `safe` or `balanced` mode and reduce intensity.
5. **"Model not found" after refactor** — confirm run from the repo root and that your checkpoint is in `models/`.

---

*Last updated: August 2026*
