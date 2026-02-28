"""Training pipeline for YOLO models."""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from ultralytics import YOLO


logger = logging.getLogger(__name__)


class TrainingPipeline:
    """YOLO model training pipeline.
    
    Orchestrates model training with validation and checkpointing.
    """
    
    def __init__(
        self,
        model: str = "yolov11n",
        device: str = "0",
        output_dir: Optional[Path] = None
    ) -> None:
        """Initialize training pipeline.
        
        Args:
            model: Model name or path ('yolov11n', 'yolov11s', path/to/model.pt')
            device: GPU device ID or 'cpu'
            output_dir: Output directory for training results
        """
        self.model_name = model
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("runs/train")
        self.model: Optional[YOLO] = None
        
        logger.info(f"Training pipeline initialized: model={model}, device={device}")
    
    def train(
        self,
        data_path: Path,
        epochs: int = 50,
        batch_size: int = 24,
        imgsz: int = 480,
        patience: int = 15,
        lr0: float = 0.002,
        **kwargs
    ) -> Dict[str, Any]:
        """Train YOLO model.
        
        Args:
            data_path: Path to dataset (data.yaml)
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Training image size
            patience: Early stopping patience
            lr0: Initial learning rate
            **kwargs: Additional arguments to YOLO.train()
            
        Returns:
            Dictionary with training results
            
        Raises:
            FileNotFoundError: If dataset not found
            RuntimeError: If training fails
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        logger.info(f"Starting training on {data_path}")
        
        try:
            # Load or create model
            self.model = YOLO(self.model_name)
            
            # Train
            results = self.model.train(
                data=str(data_path),
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                patience=patience,
                device=self.device,
                project=str(self.output_dir),
                lr0=lr0,
                **kwargs
            )
            
            logger.info(f"Training completed. Results saved to {self.output_dir}")
            
            return {
                "success": True,
                "model_path": str(self.model.trainer.best.by_id('fitness')),
                "epochs_trained": epochs,
                "final_metrics": self._extract_metrics()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}") from e
    
    def validate(self, data_path: Path) -> Dict[str, Any]:
        """Validate trained model.
        
        Args:
            data_path: Path to validation dataset
            
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train first or load existing model.")
        
        logger.info(f"Validating model on {data_path}")
        
        results = self.model.val(data=str(data_path))
        
        return {
            "map50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
            "map": float(results.box.map) if hasattr(results.box, 'map') else None,
        }
    
    def _extract_metrics(self) -> Dict[str, Any]:
        """Extract metrics from trained model."""
        if self.model is None or self.model.trainer is None:
            return {}
        
        return {
            "box_loss": float(self.model.trainer.metrics.get('train/box_loss', 0)),
            "cls_loss": float(self.model.trainer.metrics.get('train/cls_loss', 0)),
        }
