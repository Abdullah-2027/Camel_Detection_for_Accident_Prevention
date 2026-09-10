# Camel Detection & Road Segmentation for Accident Prevention

A computer vision system that detects camels on rural roads and segments the drivable road surface, aimed at preventing high-speed vehicle collisions with camels — a recurring and often fatal hazard on highways across Saudi Arabia and the wider Gulf region.

## Overview

Camels straying onto unlit rural roads are a serious cause of severe traffic accidents in the region. This project tackles the problem with two complementary computer vision models:

1. **Camel detection** — an object detector that locates camels in road scenes, the core early-warning signal.
2. **Road segmentation** — an instance-segmentation model that identifies the road surface, providing spatial context (e.g. whether a detected camel is on or approaching the drivable area).

Together they form the perception layer of what could feed a driver-alert or roadside warning system.

## Approach

The dataset combines a public object-detection/segmentation dataset with a custom-collected set of camel-and-road images, with augmentation applied to improve robustness to the lighting, distance, and terrain variation found in real road conditions.

**Camel detection**
- Model: YOLOv8n (Ultralytics), fine-tuned from pretrained weights
- Image size: 960 · Epochs: 50 · Batch: 8 · Initial LR: 0.01

**Road segmentation**
- Model: YOLO26n-seg (Ultralytics)
- Image size: 960 · Epochs: 50 · Batch: 8 · Initial LR: 0.03

Both models were trained on a single GPU.

## Results

Final validation metrics after 50 epochs.

**Camel detection (YOLOv8n)**

| Metric | Value |
|---|---|
| Precision | ~0.77 |
| Recall | ~0.82 |
| mAP@0.5 | ~0.85 |
| mAP@0.5:0.95 | ~0.56 |

**Road segmentation (YOLO26n-seg)**

| Metric | Box | Mask |
|---|---|---|
| Precision | ~0.86 | ~0.85 |
| Recall | ~0.80 | ~0.75 |
| mAP@0.5 | ~0.84 | ~0.78 |
| mAP@0.5:0.95 | ~0.58 | ~0.58 |

Both models show clean convergence — training and validation losses decrease steadily and the mAP curves plateau near the end of training, indicating a well-fit model without significant overfitting.

<!-- Replace the ~approx values above with the exact numbers from your results.csv / results.png for a polished final version. -->
<!-- Optional: add sample outputs, e.g. -->
<!-- ![Camel detection sample](assets/camel_sample.jpg) -->
<!-- ![Road segmentation sample](assets/road_sample.jpg) -->

## Tech Stack

- Python
- Ultralytics YOLO (YOLOv8n detection, YOLO26n-seg segmentation)
- PyTorch
- OpenCV

## Repository Structure

<!-- Adjust to match your actual layout -->
```
.
├── camel.yaml         # detection dataset config
├── road.yaml          # segmentation dataset config
├── train.py           # training / inference entry point
├── datasets/          # images + labels (not tracked)
├── runs/              # training outputs: weights, plots, metrics
└── README.md
```

## Getting Started

```bash
pip install ultralytics opencv-python
```

Training is driven by `train.py`, which contains a function per model:

```python
# train.py runs camel_detection() by default;
# uncomment road_segmentation() to train the segmentation model.
```

```bash
python train.py
```

Or run directly with the Ultralytics CLI:

```bash
# Camel detection
yolo train model=yolov8n.pt data=camel.yaml epochs=50 imgsz=960 batch=8 lr0=0.01

# Road segmentation
yolo train model=yolo26n-seg.pt data=road.yaml epochs=50 imgsz=960 batch=8 lr0=0.03

# Inference on new images
yolo predict model=runs/detect/train/weights/best.pt source=path/to/images conf=0.25 imgsz=960
```

## Motivation

Built as an applied computer vision project targeting a real, regionally specific safety problem. The focus was not only on accuracy but on the practical realities of the task: a limited, imbalanced dataset and conditions that mirror where these collisions actually happen.

## Author

**Abdullah Alsoghaier** — Computer Science (AI & Data Science), Prince Sultan University
[LinkedIn](https://www.linkedin.com/in/abdullah-alsoghaier) · [GitHub](https://github.com/Abdullah-2027)
