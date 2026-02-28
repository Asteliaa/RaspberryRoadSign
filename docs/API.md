# API Reference

Complete API documentation for RaspberryRoadSign.

## Core Classes

### TrafficSignDetector

High-level detection pipeline for traffic sign detection.

```python
from RaspberryRoadSign.inference.detector import TrafficSignDetector

detector = TrafficSignDetector(
    model_path="models/v1/best.pt",
    conf_threshold=0.35,
    device="auto"
)
```

**Parameters:**
- `model_path` (str | Path): Path to trained model weights
- `conf_threshold` (float): Confidence threshold (0-1), default: 0.35
- `device` (str): Device for inference ('auto', 'cpu', 'cuda'), default: 'auto'

**Methods:**

#### detect_frame()
Detect traffic signs in single frame.

```python
results = detector.detect_frame(frame)
# returns: ultralytics Results object
```

#### detect_video()
Detect traffic signs in video file.

```python
stats = detector.detect_video(
    video_path="input.mp4",
    output_path="output.mp4",
    imgsz=640
)
# returns: dict with detection statistics
```

**Returns:**
```python
{
    "total_frames": 1200,
    "detections": 450,
    "signs_detected": {
        "2.1": 150,
        "1.23": 100,
        "3.4": 50,
        ...
    }
}
```

#### get_info()
Get detector information.

```python
info = detector.get_info()
# returns: {model_info, conf_threshold, num_classes}
```

---

### YOLOWrapper

Unified wrapper around YOLOv11 model.

```python
from RaspberryRoadSign.models.yolo_wrapper import YOLOWrapper

model = YOLOWrapper(
    model_path="models/v1/best.pt",
    device="cuda"
)
```

**Methods:**

#### predict()
Run inference on source.

```python
results = model.predict(
    source=frame,
    conf=0.35,
    imgsz=640,
    iou=0.45,
    verbose=False
)
# returns: List[ultralytics.Results]
```

#### export()
Export model to different format.

```python
path = model.export(format='onnx')
# returns: Path to exported model
```

#### get_model_info()
Get model metadata.

```python
info = model.get_model_info()
# returns: {path, device, model_type, num_params}
```

---

### VideoProcessor

Process video files frame by frame.

```python
from RaspberryRoadSign.inference.video_processor import VideoProcessor

processor = VideoProcessor("input.mp4")
processor.set_output("output.mp4")

for frame in processor:
    # Process frame
    pass

processor.close()
```

**Attributes:**
- `fps`: Frames per second
- `width`: Frame width
- `height`: Frame height
- `total_frames`: Total number of frames

**Methods:**

#### set_output()
Set output video writer.

```python
processor.set_output("output.mp4", codec='mp4v')
```

#### write_frame()
Write frame to output video.

```python
processor.write_frame(annotated_frame)
```

#### close()
Release video resources.

```python
processor.close()
```

---

### Visualizer

Visualization utilities for detection results.

```python
from RaspberryRoadSign.inference.visualizer import Visualizer

visualizer = Visualizer(num_classes=155)
```

**Methods:**

#### draw_detection()
Draw detection on frame.

```python
frame = visualizer.draw_detection(
    frame=frame,
    bbox=(x1, y1, x2, y2),
    label="2.1 95%",
    class_id=0,
    alpha=0.6,
    thickness=2
)
# returns: annotated frame
```

---

### ClassMapper

Convert between RTSD class IDs and Belarusian sign codes.

```python
from RaspberryRoadSign.utils.class_mapping import ClassMapper

mapper = ClassMapper()

# ID to Belarusian code
code = mapper.id_to_belarusian(0)  # returns: "2.1"

# Validate class
is_valid = mapper.is_valid_class(0)  # returns: True

# Get all mappings
all_mappings = mapper.get_all_mappings()  # returns: {0: "2.1", 1: "1.23", ...}

# Number of classes
num_classes = mapper.get_num_classes()  # returns: 155
```

---

### Configuration Classes

Pydantic models for type-safe configuration.

```python
from RaspberryRoadSign.utils.config import InferenceConfig, TrainingConfig
```

#### InferenceConfig
```python
config = InferenceConfig(
    model_path="models/v1/best.pt",
    conf_threshold=0.35,
    iou_threshold=0.45,
    imgsz=640,
    device="cuda"
)
```

**Validation:**
- `model_path`: Must exist
- `conf_threshold`: 0.0 <= value <= 1.0
- `iou_threshold`: 0.0 <= value <= 1.0
- `imgsz`: Must be > 0

#### TrainingConfig
```python
config = TrainingConfig(
    model_path="yolov11n",
    data_path="datasets/rtsd_yolo/data.yaml",
    epochs=50,
    batch_size=24,
    imgsz=480,
    device="0",
    patience=15,
    learning_rate=0.002
)
```

---

### TrainingPipeline

YOLO model training orchestration.

```python
from RaspberryRoadSign.training.trainer import TrainingPipeline

trainer = TrainingPipeline(
    model="yolov11n",
    device="0",
    output_dir="runs/train"
)
```

**Methods:**

#### train()
Train model on dataset.

```python
results = trainer.train(
    data_path="datasets/rtsd_yolo/data.yaml",
    epochs=50,
    batch_size=24,
    imgsz=480,
    patience=15,
    lr0=0.002
)
# returns: {success, model_path, epochs_trained, final_metrics}
```

#### validate()
Validate trained model.

```python
metrics = trainer.validate("datasets/rtsd_yolo/data.yaml")
# returns: {map50, map}
```

---

## Logging

```python
from RaspberryRoadSign.utils.logging import setup_logging, get_logger

# Setup logging
logger = setup_logging(
    log_level="INFO",
    log_file="logs/app.log",
    name="MyApp"
)

# Get existing logger
logger = get_logger("MyApp")
logger.info("Training started...")
```

---

## Complete Example

```python
from RaspberryRoadSign.inference.detector import TrafficSignDetector
from RaspberryRoadSign.utils.logging import setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")

# Initialize detector
detector = TrafficSignDetector(
    model_path="models/v1/best.pt",
    conf_threshold=0.35,
    device="cuda"
)

# Process video
stats = detector.detect_video(
    video_path="input.mp4",
    output_path="output.mp4"
)

# Print results
logger.info(f"Total detections: {stats['detections']}")
logger.info(f"Detected signs: {stats['signs_detected']}")
```

---

## Type Hints

All public APIs have full type hints for IDE support and static type checking:

```python
from typing import List, Dict, Any
import numpy as np

def my_function(frame: np.ndarray, conf: float) -> List[Dict[str, Any]]:
    """Function with type hints."""
    ...
```

## Error Handling

```python
from pathlib import Path

try:
    detector = TrafficSignDetector("nonexistent.pt")
except FileNotFoundError as e:
    print(f"Model not found: {e}")

try:
    stats = detector.detect_video("input.mp4", "output.mp4")
except RuntimeError as e:
    print(f"Processing failed: {e}")
```

---

## Performance Considerations

### Memory Usage

- **yolov11n**: ~2 GB VRAM
- **yolov11s**: ~4 GB VRAM
- **yolov11m**: ~8 GB VRAM

### Inference Speed (RTX 3070)

- **480x480**: ~30 FPS
- **640x640**: ~20 FPS
- **1280x1280**: ~5 FPS

### Optimization Tips

```python
# Use smaller model for speed
detector = TrafficSignDetector("models/yolov11n.pt")

# Reduce inference size
results = detector.model.predict(frame, imgsz=416)

# Increase confidence threshold
detector.conf_threshold = 0.5

# Use CPU for memory-limited systems
detector = TrafficSignDetector(..., device="cpu")
```

---

## See Also

- [QUICKSTART](QUICKSTART.md) - Quick start guide
- [TRAINING](TRAINING.md) - Training guide
- [ARCHITECTURE](ARCHITECTURE.md) - System design
