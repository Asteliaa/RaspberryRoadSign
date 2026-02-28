# RaspberryRoadSign 🚦

Real-time traffic sign detection system using YOLOv11 for Russian and Belarusian roads.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5%2B-ee4c2c)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/ultralytics-yolov11-00457c)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

RaspberryRoadSign is a computer vision system that detects and classifies Russian traffic signs (RTSD dataset) in video streams and maps them to Belarusian traffic sign codes (GOST standard). The system is built on:

- **Model**: YOLOv11 (You Only Look Once v11)
- **Framework**: PyTorch + Ultralytics
- **Dataset**: Russian Traffic Sign Dataset (RTSD) - 155 sign classes
- **Output**: Real-time annotated video with confidence scores and sign codes

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Asteliaa/RaspberryRoadSign.git
cd RaspberryRoadSign

# Install package (development mode)
pip install -e .
```

### Run Inference

```bash
# Detect signs in video
python scripts/infer.py \
    --model models/v1/best.pt \
    --video test_video/sample.mp4 \
    --output results.mp4
```

### Train Model

```bash
# Train YOLOv11n on RTSD dataset
python scripts/train.py --config configs/training/rtsd.yaml
```

Or using make:

```bash
make infer  # Run inference on test video
make train  # Train on RTSD dataset
make test   # Run test suite
```

## Features

✅ **Real-time Detection** - Process video at 30+ FPS with GPU
✅ **155 Traffic Signs** - Russian Traffic Sign Dataset coverage
✅ **Belarusian Mapping** - Automatic conversion to GOST codes
✅ **Modular Design** - Clean API for custom integrations
✅ **Production Ready** - Type hints, logging, error handling
✅ **Documented** - Comprehensive documentation and examples

## Project Structure

```
RaspberryRoadSign/
├── src/RaspberryRoadSign/     # Main package
│   ├── models/                # YOLO model wrappers
│   ├── inference/             # Detection & visualization
│   ├── training/              # Training pipeline
│   └── utils/                 # Config, logging, mapping
├── scripts/                   # CLI entry points
├── configs/                   # Configuration files
├── tests/                     # Test suite
├── docs/                      # Documentation
└── notebooks/                 # Jupyter examples
```

## Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute tutorial
- **[INSTALLATION.md](docs/INSTALLATION.md)** - Detailed setup guide
- **[TRAINING.md](docs/TRAINING.md)** - Model training guide
- **[API.md](docs/API.md)** - Code reference
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design

## Usage Examples

### Python API

```python
from RaspberryRoadSign.inference.detector import TrafficSignDetector

# Initialize detector
detector = TrafficSignDetector(
    model_path="models/v1/best.pt",
    conf_threshold=0.35,
    device="cuda"
)

# Detect signs in video
stats = detector.detect_video(
    video_path="test_video/sample.mp4",
    output_path="results.mp4"
)

print(f"Total detections: {stats['detections']}")
print(f"Signs found: {stats['signs_detected']}")
```

### Command Line

```bash
# Basic inference
python scripts/infer.py \
    --model models/v1/best.pt \
    --video input.mp4 \
    --output output.mp4

# With custom confidence threshold
python scripts/infer.py \
    --model models/v1/best.pt \
    --video input.mp4 \
    --output output.mp4 \
    --conf 0.4

# Using GPU
python scripts/infer.py \
    --model models/v1/best.pt \
    --video input.mp4 \
    --output output.mp4 \
    --device cuda
```

## Configuration

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` to configure:
- Model path
- Confidence threshold
- CUDA device
- Output paths

See [.env.example](.env.example) for all options.

## Model Information

### Pre-trained Models

- **yolov11n.pt** (5.4 MB) - Nano model, fast inference
- **yolov11s.pt** (19 MB) - Small model, balanced
- **yolov11m.pt** (39 MB) - Medium model, high accuracy

### Trained Models

Trained model checkpoints available in `models/v1/`:
- `best.pt` - Best checkpoint by mAP
- `config.yaml` - Training configuration
- `metadata.json` - Training metrics and info

## Performance

On NVIDIA GPU (RTX 3070):
- **Inference Speed**: ~30 FPS (480x480 images)
- **Memory**: ~2 GB VRAM
- **mAP@0.5**: ~0.85 on RTSD validation set

## Testing

```bash
# Run full test suite
make test

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# With coverage report
pytest tests/ --cov=src/RaspberryRoadSign --cov-report=html
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Code formatting and linting
make format
make lint

# Type checking
mypy src/RaspberryRoadSign

# Clean up
make clean
```

## Troubleshooting

### CUDA Not Found
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Video Codec Issues
```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg           # macOS
```

### Out of Memory
```bash
# Reduce batch size or image size
python scripts/infer.py --model model.pt --video input.mp4 --output output.mp4
# Edit configs for training
```

## Dataset

This project uses the **Russian Traffic Sign Dataset (RTSD)**:
- **155 sign classes**
- **5000+ annotated images**
- **YOLO format** (txt files with normalized coordinates)

Dataset location: `datasets/rtsd_yolo/`

## References

- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [RTSD Dataset](https://www.kaggle.com/datasets/watchman/rtsd-russian-traffic-sign-dataset)
- [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

## Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## License

MIT License - see [LICENSE](LICENSE) file for details

## Citation

If you use RaspberryRoadSign in your research, please cite:

```bibtex
@software{raspberryroadsign2025,
  title={RaspberryRoadSign: Traffic Sign Detection System},
  author={RaspberryYolo Team},
  year={2025},
  url={https://github.com/Asteliaa/RaspberryRoadSign}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the team.

---

**Status**: Active Development 🚀

Last updated: Feb 28, 2025
