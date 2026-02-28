# Installation Guide

Complete setup instructions for RaspberryRoadSign.

## Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional, CPU supported)
- **Storage**: ~5GB for models and datasets
- **OS**: Linux, macOS, or Windows

## Option 1: Standard Installation (Recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/Asteliaa/RaspberryRoadSign.git
cd RaspberryRoadSign
```

### Step 2: Create Virtual Environment

Using venv:
```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

Using conda:
```bash
conda create -n raspberry-road-sign python=3.10
conda activate raspberry-road-sign
```

### Step 3: Install Package

Development install (recommended for development):
```bash
pip install -e .
```

Production install:
```bash
pip install .
```

With development tools:
```bash
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
python -c "from RaspberryRoadSign import TrafficSignDetector; print('✓ Installation successful')"
```

## Option 2: Conda Installation

If you prefer conda (environment pre-configured):

```bash
conda env create -f environment.yml
conda activate yolo_traffic
```

## GPU Setup (Optional but Recommended)

### CUDA/cuDNN Installation

**Linux (Ubuntu 20.04+)**:
```bash
# Install CUDA Toolkit
sudo apt-get install cuda-11-8

# Install cuDNN
# Download from https://developer.nvidia.com/cudnn
# Extract and copy to CUDA path
```

**macOS**: 
```bash
# GPU support via Metal Performance Shaders (MPS)
# Works automatically on Apple Silicon
```

**Windows**:
- Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Download cuDNN: https://developer.nvidia.com/cudnn
- Follow installation guides

### PyTorch with GPU Support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (MPS)
pip install torch torchvision torchaudio
```

Verify GPU detection:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Dependency Installation

### Core Dependencies

Automatically installed with package:
- torch >= 2.5.0
- ultralytics >= 8.3.0
- opencv-python >= 4.8.0
- pydantic >= 2.0.0
- pyyaml >= 6.0

### Development Dependencies

```bash
pip install -e ".[dev]"
```

Includes:
- pytest (testing)
- black (code formatting)
- pylint (linting)
- mypy (type checking)
- sphinx (documentation)

### Optional Dependencies

ONNX export support:
```bash
pip install onnxruntime-gpu  # or onnxruntime for CPU
```

## Troubleshooting Installation

### Import Error: "No module named 'torch'"

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or check: pip --version

# Reinstall PyTorch
pip install torch torchvision torchaudio
```

### CUDA Error: "CUDA out of memory"

- Use CPU: `--device cpu`
- Reduce batch size: `--batch 16`
- Use smaller model: `yolov11n` instead of `yolov11m`

### OpenCV Error: "libGL.so.1"

```bash
# Linux: Install display libraries
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Alternative: Use headless OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

### Video Codec Issues

Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

## Verify Installation

Run complete verification:

```bash
python -c "
import torch
from ultralytics import YOLO
from RaspberryRoadSign import TrafficSignDetector
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA Available:', torch.cuda.is_available())
print('✓ TrafficSignDetector imported successfully')
"
```

## Environment Configuration

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` to customize:
```
MODEL_PATH=models/v1/best.pt
CONF_THRESHOLD=0.35
DEVICE=cuda
```

## Next Steps

1. [Quick Start](QUICKSTART.md) - Run your first detection
2. [Training Guide](TRAINING.md) - Train custom models
3. [API Reference](API.md) - Code examples
4. [Architecture](ARCHITECTURE.md) - System design

## System-Specific Notes

### Apple Silicon (M1/M2/M3)

```bash
# PyTorch with Metal Performance Shaders
pip install torch torchvision torchaudio

# Use device: 'mps' for GPU acceleration
python scripts/infer.py --device mps ...
```

### Windows WSL2

```bash
# Install CUDA toolkit in WSL2
# Follow: https://docs.nvidia.com/cuda/wsl-user-guide/

# Verify GPU access
nvidia-smi
```

### Docker (Coming Soon)

```bash
docker build -t raspberry-road-sign .
docker run --gpus all -v $(pwd):/app raspberry-road-sign
```

## Getting Help

1. Check [README](../README.md) FAQ section
2. Review error messages carefully
3. Check [GitHub Issues](https://github.com/Asteliaa/RaspberryRoadSign/issues)
4. Create new issue with:
   - Error message
   - Python version (`python --version`)
   - System info (`uname -a`)
   - Installation steps you followed
