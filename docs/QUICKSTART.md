# Quick Start Guide

Get RaspberryRoadSign up and running in 5 minutes.

## Installation (2 min)

```bash
# Clone and install
git clone https://github.com/Asteliaa/RaspberryRoadSign.git
cd RaspberryRoadSign
pip install -e .
```

## Run Inference (1 min)

Detect traffic signs in a video file:

```bash
python scripts/infer.py \
    --model models/v1/best.pt \
    --video test_video/sample.mp4 \
    --output results.mp4
```

**Output**: `results.mp4` with annotated detections

## Train Model (2 min)

Fine-tune YOLOv11 on your own data:

```bash
python scripts/train.py --config configs/training/rtsd.yaml
```

## Using Python API

```python
from RaspberryRoadSign.inference.detector import TrafficSignDetector

# Load detector
detector = TrafficSignDetector("models/v1/best.pt")

# Process video
stats = detector.detect_video("input.mp4", "output.mp4")
print(f"Found {stats['detections']} detections")
```

## Using Make Commands

```bash
make install      # Install dependencies
make infer        # Run inference
make train        # Train model
make test         # Run tests
make lint         # Check code quality
make clean        # Clean build files
```

## Common Options

### Confidence Threshold
```bash
python scripts/infer.py \
    --model models/v1/best.pt \
    --video input.mp4 \
    --output output.mp4 \
    --conf 0.5  # Higher = fewer false positives
```

### GPU Selection
```bash
python scripts/infer.py \
    --model models/v1/best.pt \
    --video input.mp4 \
    --output output.mp4 \
    --device cuda:0  # or 'cpu'
```

### Training Configuration
Edit `configs/training/rtsd.yaml`:

```yaml
model: yolov11n      # yolov11n, yolov11s, yolov11m
data: datasets/rtsd_yolo/data.yaml
epochs: 50
batch_size: 24
imgsz: 480
device: 0
```

## Troubleshooting

**CUDA not found?**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Video codec error?**
```bash
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # macOS
```

**Out of memory?**
- Reduce `imgsz` (e.g., 416 instead of 480)
- Reduce `batch_size` (e.g., 16 instead of 24)

## Next Steps

1. Read [INSTALLATION.md](INSTALLATION.md) for detailed setup
2. Check [API.md](API.md) for code examples
3. See [TRAINING.md](TRAINING.md) for training guidance
4. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design

## Performance Tips

- Use GPU (`--device cuda`) for 10x speedup
- Use smaller models (yolov11n) for faster inference
- Process at lower resolution for real-time use
- Batch process videos for better throughput

Enjoy traffic sign detection! 🚦
