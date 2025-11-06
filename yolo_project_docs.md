# YOLO Object Detection Project

## Overview

This project implements a comprehensive YOLO-based object detection system for identifying tanks in images and videos. The project includes tools for model training, evaluation, testing, optimization, and deployment across multiple platforms including RKNN and TensorFlow Lite.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Key Features](#key-features)
3. [Installation & Requirements](#installation--requirements)
4. [Scripts Overview](#scripts-overview)
5. [Workflows](#workflows)
6. [Performance Optimization](#performance-optimization)
7. [Model Export & Deployment](#model-export--deployment)
8. [Usage Examples](#usage-examples)

---

## Project Structure

```
project/
├── ALL_MODEL                   # All my models
├── augmentation                # Data augmentation utilities
├── ConvertToFormat             # Export to another format
├── RKNN                        # Everything related to RKNN arch
└── TrainingAndTests            # Everything u need for training and testing
```

---

## Key Features

### 🎯 Core Capabilities

- **Multi-platform Support**: Train and deploy on CPU, GPU (CUDA), RKNN, and TensorFlow Lite
- **Advanced Detection Methods**: Standard detection, FPN (Feature Pyramid Network), and image slicing for large images
- **Comprehensive Testing**: Performance benchmarking, memory monitoring, and detailed metrics
- **Data Augmentation**: Multiple augmentation strategies including mirror, safe, balanced, and augment+
- **Model Comparison**: Tools to compare multiple trained models
- **Export Options**: Convert models to RKNN and TensorFlow Lite formats

### 📊 Monitoring & Metrics

- Real-time FPS tracking
- Memory usage monitoring (RAM and VRAM)
- Detection confidence analysis
- Frame-by-frame logging (CSV format)
- Visualization of training metrics
- Performance graphs and histograms

---

## Installation & Requirements

### Prerequisites

```bash
pip install ultralytics opencv-python torch numpy pandas matplotlib psutil pillow
```

### Optional Dependencies

```bash
# For RKNN export
pip install rknn-toolkit2

# For TensorFlow Lite
pip install tensorflow
```

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with CUDA support, 16GB+ RAM
- **For RKNN**: Rockchip RK3588/RK3576/RK3566 series boards

---

## Scripts Overview

### 1. Model Training & Evaluation

#### `model_train.py`

Trains YOLO models with customizable hyperparameters.

**Key Features:**
- Configurable training parameters (epochs, batch size, learning rate)
- Multiple optimizer support (SGD, AdamW)
- Early stopping with patience
- Automatic model evaluation on test set
- Confusion matrix generation

**Parameters:**
```python
epochs=100          # Number of training epochs
batch_size=16       # Batch size for training
img_size=640        # Input image size
lr0=0.001          # Initial learning rate
lrf=0.001          # Final learning rate
optimizer='SGD'     # Optimizer type
momentum=0.937      # Momentum for SGD
patience=10         # Early stopping patience
```

#### `test_several_models.py`

Compares multiple trained models side-by-side.

**Output Metrics:**
- mAP@0.5
- mAP@0.5:0.95
- mAP@0.75
- Per-class performance

---

### 2. Testing & Benchmarking

#### `model_test.py`

Comprehensive testing script with advanced monitoring.

**Features:**
- Image and video processing
- Frame skipping for faster processing
- Individual frame saving with detections
- Memory monitoring (RAM and VRAM)
- CSV logging of all metrics
- Performance visualization (FPS plots, confidence histograms)
- Warmup iterations for stable measurements

**Command Line Options:**
```bash
python model_test.py --source video.mp4 --model best.pt \
    --conf 0.25 --iou 0.45 --device 0 \
    --frame-skip 5 --save-results --save-frames \
    --memory-monitor
```

**Output:**
- Processed video with annotations
- Individual frames (optional)
- CSV log file with per-frame metrics
- FPS performance graphs
- Confidence distribution histogram

#### `opt_test_model.py`

Optimized testing for maximum performance (no visualization).

**Features:**
- Minimal overhead for pure speed testing
- GPU optimization with cudnn.benchmark
- Statistical analysis (mean, min, max, std)
- Real-time coefficient calculation

#### `benchmark_yolo_video.py`

Dedicated benchmarking tool for accurate performance measurement.

**Features:**
- GPU synchronization for accurate timing
- Warmup period for stable results
- Separate inference and frame processing timing
- Tests on both videos and synthetic images
- Multiple resolution testing

---

### 3. Advanced Detection

#### `fpn-detection.py`

Multi-scale detection using Feature Pyramid Network approach.

**Features:**
- Multi-scale inference (e.g., 0.5x, 1.0x, 1.5x)
- Large image slicing with overlap
- NMS across multiple scales
- Color-coded detections by scale
- Supports images, videos, and webcam

**Command Line Options:**
```bash
python fpn-detection.py --source video.mp4 --model best.pt \
    --use-fpn --fpn-scales 0.5,1.0,1.5 \
    --process-large --slice-size 640 --overlap 0.2 \
    --save-results --show
```

**Processing Modes:**
- **Standard**: Single-scale detection
- **FPN**: Multi-scale detection for better small object detection
- **Slicing**: Large image processing by dividing into overlapping tiles

---

### 4. Data Preprocessing

#### `augmentation.py`

Advanced data augmentation toolkit.

**Augmentation Types:**

1. **Mirror** (`--type mirror`): Simple horizontal flip
2. **Safe** (`--type safe`): Minimal artifacts
   - Hue shift (reduced range)
   - Noise (reduced amount)
   - Light brightness/exposure changes
3. **Balanced** (`--type balanced`): 2-3 augmentations from different groups
   - One color augmentation (hue/brightness/exposure)
   - Optional spatial augmentation (crop/shift)
   - Optional noise
4. **Augment+** (`--type augmentplus`): Random 3-5 augmentations
   - Crop, rotation, shift
   - Hue shift, brightness, exposure
   - Noise

**Usage:**
```bash
python augmentation.py ./images --output ./augmented \
    --type augmentplus --min-augs 3 --max-augs 5
```

**Features:**
- Sequential numbering continuation
- Maintains original image dimensions
- Duplicate detection and prevention
- Configurable noise levels

#### `cat_vid.py`

Extract frames from videos at specified intervals.

**Usage:**
```bash
python cat_vid.py
# Configure video_file, output_dir, and interval in the script
```

**Features:**
- Configurable frame extraction interval
- Progress reporting
- Video information display
- Batch processing support

#### `resize_images.py`, `resize_and_crop.py`, `resize_and_pad.py`

Image preprocessing utilities for standardizing input sizes.

**Options:**
- **resize_and_pad.py**: Maintains aspect ratio, adds black padding
- **resize_and_crop.py**: Maintains aspect ratio, crops to fit
- **resize_images.py**: Batch processing with padding

---

### 5. Visualization & Analysis

#### `Metric_graph.py`

Visualize training metrics from CSV files.

**Usage:**
```python
plot_accuracy_from_csv("results.csv", metric='metrics/mAP50(B)')
```

**Supported Metrics:**
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Loss curves

---

### 6. Model Export & Deployment

#### `ConvertToFormat_rknn.py`

Export YOLO models to RKNN format for Rockchip NPUs.

**Supported Targets:**
- rk3588, rk3576, rk3566, rk3568, rk3562
- rv1103, rv1106, rv1103b, rv1106b, rk2118

**Usage:**
```python
model = YOLO("best_yolo11n.pt")
model.export(format="rknn", name="rk3588")
```

#### `ConvertToFormat_TF_lite.py`

Export to TensorFlow Lite for mobile and embedded devices.

**Output:**
- Float32 model for maximum compatibility
- Optimized for inference on mobile devices

#### `test_rknn_model_Image.py` / `test_rknn_model_Video.py`

Test RKNN-converted models on images and videos.

**Output Formats:**
- Annotated images/videos
- JSON detection results
- Text summaries
- Statistics and metrics

---

## Workflows

### 1. Complete Training Pipeline

```bash
# 1. Prepare data
python cat_vid.py  # Extract frames from videos
python augmentation.py ./frames --type augmentplus  # Augment data
python resize_images.py ./frames --output ./processed  # Standardize sizes

# 2. Train model
python model_train.py  # Edit parameters in script

# 3. Visualize training
python Metric_graph.py  # Plot metrics

# 4. Evaluate and compare
python test_several_models.py  # Compare multiple models
```

### 2. Testing & Benchmarking Pipeline

```bash
# 1. Basic testing
python model_test.py --source video.mp4 --model best.pt --save-results

# 2. Performance benchmarking
python benchmark_yolo_video.py  # Edit paths in script

# 3. Optimized speed test
python opt_test_model.py  # Edit paths in script

# 4. Advanced FPN testing
python fpn-detection.py --source video.mp4 --model best.pt \
    --use-fpn --fpn-scales 0.5,1.0,1.5 --save-results
```

### 3. Deployment Pipeline

```bash
# 1. Export to RKNN
python ConvertToFormat_rknn.py

# 2. Test RKNN model
python test_rknn_model_Image.py
python test_rknn_model_Video.py

# 3. Export to TensorFlow Lite
python ConvertToFormat_TF_lite.py
```

---

## Performance Optimization

### Tips for Maximum Speed

1. **Hardware Optimization**
   - Use CUDA-enabled GPU
   - Enable cudnn.benchmark
   - Use appropriate batch sizes

2. **Processing Optimization**
   - Use frame skipping for videos (`--frame-skip`)
   - Disable visualization during benchmarking
   - Use optimized inference mode

3. **Model Optimization**
   - Use smaller models (YOLOv11n) for speed
   - Lower input resolution if acceptable
   - Adjust confidence threshold

### Typical Performance Metrics

**GPU (NVIDIA RTX 3060):**
- 640x640 images: ~50-70 FPS
- 1920x1080 video: ~30-40 FPS
- FPN mode: ~10-15 FPS

**CPU (Modern i7):**
- 640x640 images: ~5-10 FPS
- 1920x1080 video: ~2-5 FPS

**RKNN (RK3588):**
- 640x640 images: ~30-50 FPS
- Optimized for embedded deployment

---

## Usage Examples

### Training a Model

```python
# Edit model_train.py parameters
data_yaml_path = "./data.yaml"
results = train_yolo(
    data_yaml_path=data_yaml_path,
    epochs=100,
    batch_size=16,
    img_size=640,
    experiment_name='tank_detector_v1',
    lr0=0.001,
    lrf=0.001
)
```

### Testing on Video with Full Monitoring

```bash
python model_test.py \
    --source ./test_video.mp4 \
    --model ./best.pt \
    --conf 0.25 \
    --iou 0.45 \
    --device 0 \
    --frame-skip 5 \
    --save-results \
    --save-frames \
    --frames-dir ./output_frames \
    --memory-monitor \
    --show
```

### Multi-Scale FPN Detection

```bash
python fpn-detection.py \
    --source ./large_image.jpg \
    --model ./best.pt \
    --use-fpn \
    --fpn-scales 0.5,1.0,1.5,2.0 \
    --conf 0.25 \
    --iou 0.45 \
    --save-results
```

### Data Augmentation

```bash
# Balanced augmentation (recommended)
python augmentation.py ./training_images \
    --output ./augmented_images \
    --type balanced

# Aggressive augmentation
python augmentation.py ./training_images \
    --output ./augmented_images \
    --type augmentplus \
    --min-augs 4 \
    --max-augs 6 \
    --noise 0.08
```

---

## Output Files & Logs

### Model Testing Outputs

**CSV Logs** (`logs/video_log_TIMESTAMP.csv`):
```csv
Timestamp,Frame,ProcessingTime,InstantFPS,Detections,AvgConfidence,MinConfidence,MaxConfidence,AvgBoxArea,FrameRes,InputRes,SkippedFrames
```

**Performance Graphs**:
- `fps_plot.png`: FPS over time
- `confidence_histogram.png`: Distribution of detection confidence

**Video Results**:
- Annotated video with bounding boxes
- Frame-by-frame saved images (optional)

### RKNN Testing Outputs

**JSON Format**:
```json
{
  "video_path": "input.mp4",
  "timestamp": "20250101_120000",
  "summary": {
    "total_detections": 1234,
    "detection_stats": {"tank": 1234},
    "avg_detections_per_frame": 2.5
  },
  "frame_detections": [...]
}
```

---

## Configuration Tips

### CUDA Setup

```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
```

### Memory Management

```python
# Clear CUDA cache
import torch
torch.cuda.empty_cache()

# Enable memory monitoring
python model_test.py --memory-monitor
```

### Optimal Hyperparameters

**For Tank Detection:**
- Confidence threshold: 0.25-0.35
- IoU threshold: 0.45-0.7
- Input size: 640x640
- Batch size: 16 (adjust based on GPU memory)

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Lower input resolution
   - Enable frame skipping

2. **Slow Performance**
   - Check GPU utilization
   - Enable cudnn.benchmark
   - Use optimized testing script

3. **Poor Detection Quality**
   - Adjust confidence threshold
   - Use FPN for small objects
   - Increase training epochs

4. **Augmentation Artifacts**
   - Use "safe" or "balanced" mode
   - Reduce augmentation intensity
   - Check input image quality

---

## Project Goals

This project aims to provide a complete, production-ready object detection system with:

- ✅ High accuracy tank detection
- ✅ Real-time processing capabilities
- ✅ Multi-platform deployment support
- ✅ Comprehensive testing and monitoring tools
- ✅ Flexible data augmentation pipeline
- ✅ Performance optimization utilities

---

## License & Credits

This project uses:
- **Ultralytics YOLO** for object detection
- **OpenCV** for image/video processing
- **PyTorch** for deep learning
- **RKNN Toolkit** for NPU deployment

---

## Future Enhancements

- [ ] Real-time tracking with object IDs
- [ ] Multi-camera support
- [ ] Cloud deployment options
- [ ] Web interface for monitoring
- [ ] Automated hyperparameter tuning
- [ ] Model quantization for faster inference

---

*Last Updated: November 2025*