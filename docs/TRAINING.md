# Training Guide

Train YOLOv11 models on custom traffic sign datasets.

## Quick Start

Train on RTSD dataset with default config:

```bash
python scripts/train.py --config configs/training/rtsd.yaml
```

Results saved to `runs/train/`

## Configuration

All training parameters in YAML config file.

Model sizes: yolov11n, yolov11s, yolov11m

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| yolov11n | 5.4M | Fastest | Good | 2GB |
| yolov11s | 19M | Fast | Better | 4GB |
| yolov11m | 39M | Slow | Best | 8GB |

## Training Process

1. Prepare dataset in YOLO format
2. Edit configs/training/rtsd.yaml
3. Run: python scripts/train.py --config configs/training/rtsd.yaml
4. Monitor training progress
5. Best model saved to runs/train/*/weights/best.pt

## Dataset Format

```
datasets/rtsd_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## Common Issues

### Out of Memory
- Reduce batch_size: 24 -> 16
- Reduce imgsz: 480 -> 416
- Use smaller model: yolov11n

### Training Too Slow
- Reduce imgsz: 480 -> 416
- Increase batch_size: 24 -> 32
- Use GPU (not CPU)

### Model Not Improving
- Train longer: epochs 50 -> 100
- Increase learning_rate: 0.002 -> 0.005
- Use larger model: yolov11s

### Poor Accuracy
- Increase imgsz: 480 -> 640
- Use larger model: yolov11m
- Train longer: epochs 50 -> 100

## See Also

- QUICKSTART.md - Quick start guide
- API.md - Code reference
- ARCHITECTURE.md - System design
