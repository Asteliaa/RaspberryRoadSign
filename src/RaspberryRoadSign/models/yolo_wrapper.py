"""YOLO model wrapper - Unified interface for YOLOv11 models."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class YOLOWrapper:
    """Unified wrapper around YOLOv11 model for consistent interface.
    
    This class abstracts away YOLO-specific details and provides a clean
    interface for loading, predicting, and exporting models.
    
    Attributes:
        model_path: Path to model weights file
        device: Device to run inference on ('cpu', 'cuda', 'mps', 'auto')
        model: Loaded YOLO model instance
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'auto'
    ) -> None:
        """Initialize YOLO model wrapper.
        
        Args:
            model_path: Path to .pt model weights file
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
            
        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If model fails to load
        """
        self.model_path = Path(model_path)
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            self.model: YOLO = YOLO(str(self.model_path))
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_path}: {e}") from e
    
    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: float = 0.35,
        imgsz: int = 640,
        iou: float = 0.45,
        verbose: bool = False
    ) -> List[Results]:
        """Run inference on source.
        
        Args:
            source: Image/frame, video path, or numpy array
            conf: Confidence threshold (0-1)
            imgsz: Inference image size
            iou: IoU threshold for NMS
            verbose: Whether to print verbose output
            
        Returns:
            List of YOLO Results objects
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            imgsz=imgsz,
            iou=iou,
            verbose=verbose
        )
        return results
    
    def export(
        self,
        format: str = 'onnx',
        output_path: Optional[Path] = None
    ) -> Path:
        """Export model to specified format.
        
        Args:
            format: Export format ('onnx', 'tflite', 'torchscript', etc.)
            output_path: Optional output path for exported model
            
        Returns:
            Path to exported model
        """
        if output_path is None:
            output_path = self.model_path.parent
        
        exported = self.model.export(format=format, imgsz=640)
        return Path(exported)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "path": str(self.model_path),
            "device": self.device,
            "model_type": self.model.model.__class__.__name__,
            "num_params": sum(p.numel() for p in self.model.model.parameters()),
        }
