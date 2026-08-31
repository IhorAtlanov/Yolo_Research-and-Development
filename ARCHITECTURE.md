# Architecture & Contribution Guidelines

This document defines **where new code belongs** and the conventions to follow
so the repository stays clean, modular, and maintainable. **Read this before
adding any file to the repository.**

---

## 1. Design Principles

The layout follows a few simple rules:

1. **Separation of Concerns (SoC)** — code is grouped by *domain responsibility*, not by file type or arbitrary folder.
2. **Source lives under `src/`** — no runnable Python at the repository root.
3. **Binaries are not versioned** — model artifacts, datasets, and generated outputs are gitignored.
4. **Configuration is isolated** — configs have their own home (`config/`), never live next to source.
5. **Tests mirror `src/`** — tests live in `tests/` and mirror the package hierarchy.
6. **Consistent naming** — see [Naming conventions](#4-naming-conventions).

---

## 2. Where Things Go

### `src/` — Python source (importable package)

Each sub-package is a **domain module**:

| Directory | Responsibility | Existing modules |
| --- | --- | --- |
| `src/training/` | Model training, hyperparameters, training loop | `model_train.py` |
| `src/inference/` | Running trained models: testing, benchmarking, FPN | `model_test.py`, `opt_test_model.py`, `benchmark_yolo_video.py`, `fpn_detection.py` |
| `src/evaluation/` | Comparing models, metrics, plotting, environment checks | `test_several_models.py`, `metric_graph.py`, `cuda_test.py` |
| `src/data/` | Data preparation: augmentation, frame extraction, resizing | `augmentation.py`, `cat_vid.py`, `resize_*.py` |
| `src/export/` | Converting trained models to deployment formats | `convert_to_format_rknn.py`, `convert_to_format_tflite.py` |
| `src/deployment/rknn/` | Runtime testing of RKNN-deployed models | `test_rknn_model_image.py`, `test_rknn_model_video.py` |

**Where to add new code:**

- New **inference method** (e.g. a new inference ONNX runtime) → `src/inference/`
- New **training strategy** (e.g. transfer-learning variants) → `src/training/`
- New **data transform** → `src/data/`
- New **deployment target** (e.g. TensorRT server, CoreML, OpenVINO) → create a new module under `src/deployment/<target>/`
- New **export backend** → `src/export/`

> If you add a module that belongs to an existing domain, add it to the matching
> `src/<domain>/` package. If it spans a **new** domain, create a new top-level
> package under `src/` and add the `__init__.py`.

### `models/` — model artifacts (gitignored)

Trained weights and converted models, grouped by format:

```
models/
├── best_yolo11n.pt      # PyTorch checkpoint
├── onnx/                # ONNX + TensorRT engine
├── tflite/              # TensorFlow Lite (SavedModel + .tflite)
└── rknn/                # RKNN converted models + metadata
```

- **Never commit** model binaries — add a `.gitkeep` if the folder must exist in git.
- File naming: `<model_name>.<ext>` (e.g. `best_yolo11n.pt`, `best_yolo11n-rk3588.rknn`).

### `config/` — configuration templates

- Non-secret configuration **templates** go here (e.g. `data.yaml.template`).
- Copy-to-use pattern: `cp config/<name>.template <destination>`.
- Keep secrets out of git; use ignore files or local config.

### `data/` — datasets & media (gitignored)

- Raw images/videos, extracted frames, augmented output, and datasets.
- Add a `.gitkeep` if you need the directory tracked; **do not commit datasets**.

### `outputs/` — runtime artifacts (gitignored)

- Results, annotated media, logs, CSV, plots, and benchmarking output.
- Scripts that generate output should write here (or into their own `output_dir`).

### `tests/` — automated tests

- Unit/integration tests, **mirroring the `src/` hierarchy**:
  - `tests/training/test_model_train.py` → tests `src/training/model_train.py`
  - `tests/data/test_augmentation.py` → tests `src/data/augmentation.py`
- Add a `__init__.py` in each test sub-package.

### `scripts/` — ops / maintenance scripts

- One-off or automation scripts that are **not** part of the runtime package
  (e.g. `migrate_to_src_layout.sh`).
- Prefer shell, or a thin Python wrapper that imports from `src/`.

### `docs/`

- Long-form documentation beyond the README (this file lives at `docs/../ARCHITECTURE.md`).
- Guidance, decision records (ADRs), and design notes.

### `requirements/` — split dependency groups

- `base.txt` — required for all core usage.
- `rknn.txt`, `tflite.txt` — platform-specific extras.
- `dev.txt` — linting/testing tooling.

---

## 3. Placement Quick-Check

| "I want to add a..." | Put it in... |
| --- | --- |
| New training script | `src/training/` |
| New inference/benchmark script | `src/inference/` |
| New evaluation/comparison tool | `src/evaluation/` |
| New augmentation/resize util | `src/data/` |
| New model format converter | `src/export/` |
| Test for a source module | `tests/<domain>/test_<module>.py` |
| Dataset config template | `config/` |
| Model weights / converted model | `models/` (gitignored) |
| Benchmark results, logs, plots | `outputs/` (gitignored) |
| Datasets, raw media | `data/` (gitignored) |
| One-off maintenance script | `scripts/` |

---

## 4. Naming Conventions

- **Files/dirs**: `snake_case` for Python files (renamed legacy files follow this rule, e.g. `Metric_graph.py` → `metric_graph.py`, `fpn-detection.py` → `fpn_detection.py`).
- **Functions/variables**: `snake_case`.
- **Classes**: `PascalCase`.
- **Constants**: `UPPER_SNAKE_CASE`.

## 5. Path Handling

Scripts are run from the **repository root**. Relative paths into `models/`,
`data/`, and `outputs/` assume that. When importing between modules, prefer
absolute imports relative to the `src/` package (e.g. `from src.data import
augmentation`) and add the repo root to `PYTHONPATH` if needed.

> **Important:** if a script *moves* directories, its relative paths must be
> re-verified. The current module depths are:
> - `src/*.py` → `../models/`, `../data/`, `../outputs/`
> - `src/deployment/rknn/*.py` → `../../../models/...`

## 6. Quality Gates

- New source modules should be covered by a unit test under `tests/`.
- Deprecated/legacy flat folders (`TrainingAndTests/`, `ALL_MODEL/`, etc.) are **removed** — do not recreate files at the repo root.

---

## 7. Roadmap (proposed enhancements)

- [ ] Standardize model/data/output paths via environment variables or a shared settings module (`src/core/config.py`).
- [ ] Convert flat `__main__`-style scripts into reusable functions with `if __name__ == "__main__"` guards and consistent CLI (`argparse`).
- [ ] Add `pytest` coverage targets and a CI workflow (lint via `ruff`).
- [ ] Add TensorRT server and OpenVINO deployment modules under `src/deployment/`.
