# System Architecture

Overview of RaspberryRoadSign system design and components.

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                  INPUT                               │
│         (Video File / Video Stream / Frame)          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            VIDEO PROCESSOR                           │
│  • Frame extraction from video                       │
│  • Frame buffering & management                      │
│  • Output video writing                              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           TRAFFIC SIGN DETECTOR                      │
│  • Orchestrates detection pipeline                   │
│  • Frame-by-frame inference                          │
│  • Detection aggregation                             │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐   ┌──────────────────┐
│  YOLO WRAPPER    │   │  CLASS MAPPER    │
│  • Model loading │   │  • RTSD ID → BY  │
│  • Inference     │   │  • Code mapping  │
│  • Export        │   └──────────────────┘
└────────┬─────────┘
         │
         ▼
    [YOLO Model]
         │
         ▼
┌──────────────────┐
│  DETECTIONS      │
│  • Boxes         │
│  • Confidence    │
│  • Class IDs     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│             VISUALIZER                               │
│  • Draw bounding boxes                               │
│  • Annotate with sign codes                          │
│  • Confidence score display                          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│             OUTPUT                                   │
│      (Annotated Video with Detections)              │
└──────────────────────────────────────────────────────┘
```

## Module Organization

### `src/RaspberryRoadSign/` - Main Package

#### `models/` - Model Management
```
models/
├── yolo_wrapper.py      # YOLOv11 wrapper with unified interface
└── types.py            # Type definitions for models
```

**Responsibility**: Abstract YOLO model details, provide consistent inference interface.

**Key Classes**:
- `YOLOWrapper` - Load, predict, export YOLO models

---

#### `inference/` - Detection Pipeline
```
inference/
├── detector.py         # High-level detection orchestrator
├── video_processor.py  # Video file I/O
├── visualizer.py       # Annotation and drawing
└── types.py           # Inference type definitions
```

**Responsibility**: Process videos, run detection, visualize results.

**Key Classes**:
- `TrafficSignDetector` - Main detection pipeline
- `VideoProcessor` - Handle video file operations
- `Visualizer` - Draw results on frames

---

#### `training/` - Training Pipeline
```
training/
├── trainer.py          # Training orchestration
└── callbacks.py        # Custom training callbacks
```

**Responsibility**: Handle model training with full automation.

**Key Classes**:
- `TrainingPipeline` - Train and validate models

---

#### `utils/` - Utilities
```
utils/
├── config.py           # Pydantic configuration models
├── logging.py          # Logging setup
├── class_mapping.py    # RTSD → Belarusian code mapping
├── paths.py           # Path utilities
└── types.py           # Shared type definitions
```

**Responsibility**: Provide utilities for configuration, logging, class mapping.

---

#### `data/` - Data Handling (Future)
```
data/
├── dataset.py          # Dataset loading
├── transforms.py       # Data augmentation
└── types.py           # Data type definitions
```

---

### `scripts/` - CLI Entry Points
```
scripts/
├── infer.py           # Inference command-line script
├── train.py           # Training command-line script
└── convert_dataset.py # Dataset format conversion
```

**Purpose**: User-friendly CLI interfaces to main functionality.

---

### `tests/` - Test Suite
```
tests/
├── conftest.py        # Pytest fixtures and configuration
├── unit/
│   ├── test_mapping.py      # Class mapping tests
│   ├── test_config.py       # Configuration tests
│   └── test_utils.py        # Utility function tests
├── integration/
│   ├── test_inference.py    # End-to-end inference tests
│   └── test_training.py     # Training pipeline tests
└── fixtures/
    ├── sample_video.mp4
    └── sample_frame.jpg
```

---

### `docs/` - Documentation
```
docs/
├── README.md          # Main documentation
├── QUICKSTART.md      # 5-minute guide
├── INSTALLATION.md    # Setup instructions
├── TRAINING.md        # Training guide
├── API.md            # Code reference
└── ARCHITECTURE.md    # This document
```

---

## Data Flow

### Inference Pipeline
```
Input Video
    ↓
[VideoProcessor.read_frame()]
    ↓
Frame (numpy array)
    ↓
[TrafficSignDetector.detect_frame()]
    ↓
[YOLOWrapper.predict()]
    ↓
YOLO Results
    ↓
[ClassMapper.id_to_belarusian()]
    ↓
Detections with Belarusian codes
    ↓
[Visualizer.draw_detection()]
    ↓
Annotated Frame
    ↓
[VideoProcessor.write_frame()]
    ↓
Output Video
```

### Training Pipeline
```
Training Config (YAML)
    ↓
[load_config()]
    ↓
Config Dict
    ↓
[TrainingPipeline.train()]
    ↓
[YOLO.train()]
    ↓
Training Loop (50 epochs)
    ↓
Checkpoints & Metrics
    ↓
Best Model
    ↓
models/v1/best.pt
```

## Class Relationships

```python
# Main entry point
TrafficSignDetector
├── uses: YOLOWrapper (for inference)
├── uses: VideoProcessor (for video I/O)
├── uses: Visualizer (for annotation)
├── uses: ClassMapper (for ID → code mapping)
└── uses: InferenceConfig (for configuration)

TrainingPipeline
├── uses: YOLO (from ultralytics)
├── uses: TrainingConfig (for configuration)
└── produces: trained model weights

# Configuration
InferenceConfig (Pydantic BaseModel)
├── validates: model_path
├── validates: conf_threshold
└── validates: device

TrainingConfig (Pydantic BaseModel)
├── validates: model_path
├── validates: data_path
└── validates: training parameters
```

## Design Patterns

### 1. **Dependency Injection**
Configuration objects passed to constructors:
```python
detector = TrafficSignDetector(
    model_path="models/v1/best.pt",
    conf_threshold=0.35
)
```

### 2. **Strategy Pattern**
VideoProcessor handles different video codecs via codec parameter:
```python
processor.set_output("output.mp4", codec='mp4v')
```

### 3. **Factory Pattern**
Visualizer creates color schemes based on num_classes:
```python
visualizer = Visualizer(num_classes=155)
```

### 4. **Adapter Pattern**
YOLOWrapper adapts Ultralytics YOLO to consistent interface:
```python
# Instead of: model.predict(...)[0].boxes
# We provide: wrapper.predict(...)
```

### 5. **Template Method**
VideoProcessor iteration pattern:
```python
for frame in processor:
    # process frame
```

## Configuration Management

### YAML-based Training Config
```yaml
# configs/training/rtsd.yaml
model: yolov11n
data: datasets/rtsd_yolo/data.yaml
epochs: 50
batch_size: 24
device: 0
```

### Pydantic Validation
```python
# Type-safe, validated at runtime
config = TrainingConfig.from_yaml('configs/training/rtsd.yaml')
# Raises ValidationError if invalid
```

### Environment Variables
```python
# .env file support
model_path = os.getenv('MODEL_PATH', 'models/v1/best.pt')
conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.35'))
```

## Error Handling

**Three-tier error strategy:**

1. **Type Validation** (Pydantic)
   - Invalid config → ValidationError
   
2. **Runtime Checks** (try-except)
   - File not found → FileNotFoundError
   - Model loading fails → RuntimeError
   
3. **Logging** (structured logging)
   - All errors logged with context
   - Stack traces preserved for debugging

```python
try:
    detector = TrafficSignDetector("nonexistent.pt")
except FileNotFoundError as e:
    logger.error(f"Model not found: {e}")
    raise
```

## Performance Considerations

### Memory Optimization
- **Frame buffering**: Single frame in memory at time
- **Lazy loading**: Models loaded only when needed
- **Batch processing**: Future support for batched inference

### Speed Optimization
- **GPU acceleration**: CUDA/cuDNN support
- **Vectorization**: NumPy operations
- **Caching**: Color arrays pre-computed

### Scalability
- **Streaming**: Frame-by-frame processing (low memory)
- **Async**: Future support for async inference
- **Distributed**: Future support for multi-GPU

## Testing Strategy

### Unit Tests
- Individual function behavior
- Config validation
- Class mapping accuracy

### Integration Tests
- End-to-end inference
- Video processing pipeline
- Training workflow

### Smoke Tests
- Model loading
- Library imports
- CUDA availability

## Extension Points

### Adding New Detection Models
1. Create new class inheriting from `YOLOWrapper`
2. Override `predict()` method
3. Update TrafficSignDetector to support

### Custom Visualizations
1. Subclass `Visualizer`
2. Override `draw_detection()` method
3. Add custom drawing logic

### Training Callbacks
1. Create callback class
2. Inherit from YOLO callbacks
3. Register with TrainingPipeline

## Dependencies

### Core
- **torch** - Deep learning framework
- **ultralytics** - YOLO implementation
- **opencv-python** - Computer vision library
- **pydantic** - Data validation
- **pyyaml** - Configuration parsing

### Development
- **pytest** - Testing framework
- **black** - Code formatting
- **pylint** - Code linting
- **mypy** - Type checking
- **sphinx** - Documentation generation

## Future Improvements

1. **Async Inference** - Non-blocking detection
2. **Model Quantization** - Reduce model size
3. **Real-time Camera** - Webcam support
4. **REST API** - Web service wrapper
5. **Docker Containerization** - Easy deployment
6. **Multi-GPU Support** - Parallel processing
7. **Model Ensemble** - Combine multiple models
8. **Confidence Calibration** - Improve accuracy metrics

---

See also:
- [API Reference](API.md) - Detailed API documentation
- [Training Guide](TRAINING.md) - How to train models
- [Quick Start](QUICKSTART.md) - Getting started
