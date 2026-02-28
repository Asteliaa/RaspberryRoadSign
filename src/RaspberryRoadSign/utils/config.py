"""Configuration models for training and inference."""

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class InferenceConfig(BaseModel):
    """Configuration for inference/detection pipeline.
    
    Attributes:
        model_path: Path to trained model weights
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
        imgsz: Inference image size
        device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
    """
    
    model_path: Path = Field(
        ...,
        description="Path to model weights file (.pt)"
    )
    conf_threshold: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections"
    )
    iou_threshold: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS"
    )
    imgsz: int = Field(
        640,
        gt=0,
        description="Inference image size"
    )
    device: str = Field(
        "auto",
        description="Device for inference"
    )
    
    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: Path) -> Path:
        """Validate that model path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {v}")
        return path
    
    class Config:
        """Pydantic config."""
        env_file = '.env'
        env_prefix = ''


class TrainingConfig(BaseModel):
    """Configuration for model training.
    
    Attributes:
        model_path: Base model to train (checkpoint or pretrained)
        data_path: Path to dataset in YOLO format
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Training image size
        device: Device for training
        patience: Early stopping patience
        learning_rate: Initial learning rate
    """
    
    model_path: Path = Field(
        ...,
        description="Base model path or model name"
    )
    data_path: Path = Field(
        ...,
        description="Path to YOLO format dataset"
    )
    epochs: int = Field(
        50,
        gt=0,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        24,
        gt=0,
        description="Batch size"
    )
    imgsz: int = Field(
        480,
        gt=0,
        description="Training image size"
    )
    device: str = Field(
        "0",
        description="Device for training (GPU ID or 'cpu')"
    )
    patience: int = Field(
        15,
        ge=0,
        description="Early stopping patience (epochs)"
    )
    learning_rate: float = Field(
        0.002,
        gt=0,
        description="Initial learning rate"
    )
    output_dir: Optional[Path] = Field(
        None,
        description="Output directory for results"
    )
    
    @field_validator('data_path')
    @classmethod
    def validate_data_path(cls, v: Path) -> Path:
        """Validate that data path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Data path does not exist: {v}")
        return path
    
    class Config:
        """Pydantic config."""
        env_file = '.env'
        env_prefix = ''


class DatasetConfig(BaseModel):
    """Configuration for dataset paths.
    
    Attributes:
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Optional path to test images
    """
    
    train_path: Path = Field(..., description="Training images path")
    val_path: Path = Field(..., description="Validation images path")
    test_path: Optional[Path] = Field(None, description="Test images path")
    
    @field_validator('train_path', 'val_path', 'test_path')
    @classmethod
    def validate_paths(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate dataset paths exist."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Path does not exist: {v}")
            return path
        return v
