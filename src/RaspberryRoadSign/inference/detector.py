"""High-level traffic sign detection pipeline."""

from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging
import numpy as np
from ultralytics.engine.results import Results

from RaspberryRoadSign.models.yolo_wrapper import YOLOWrapper
from RaspberryRoadSign.utils.class_mapping import ClassMapper
from RaspberryRoadSign.inference.video_processor import VideoProcessor
from RaspberryRoadSign.inference.visualizer import Visualizer


logger = logging.getLogger(__name__)


class TrafficSignDetector:
    """High-level traffic sign detection pipeline.
    
    Orchestrates model loading, video processing, and visualization
    for traffic sign detection.
    
    Attributes:
        model: YOLOWrapper instance
        class_mapper: ClassMapper for ID to code conversion
        conf_threshold: Confidence threshold for detections
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.35,
        device: str = 'auto'
    ) -> None:
        """Initialize detector.
        
        Args:
            model_path: Path to trained model weights
            conf_threshold: Confidence threshold (0-1)
            device: Device for inference
            
        Raises:
            FileNotFoundError: If model not found
            RuntimeError: If model fails to load
        """
        self.model = YOLOWrapper(model_path, device=device)
        self.class_mapper = ClassMapper()
        self.conf_threshold = conf_threshold
        logger.info(f"Detector initialized with model: {model_path}")
    
    def detect_frame(self, frame: np.ndarray) -> Optional[Results]:
        """Detect traffic signs in single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            YOLO Results object or None if detection fails
        """
        try:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                verbose=False
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return None
    
    def detect_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        imgsz: Optional[int] = None
    ) -> Dict[str, Any]:
        """Detect traffic signs in video file.
        
        Args:
            video_path: Input video path
            output_path: Output annotated video path
            imgsz: Inference image size (auto if None)
            
        Returns:
            Dictionary with detection statistics
            
        Raises:
            FileNotFoundError: If video not found
            RuntimeError: If video processing fails
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        processor = VideoProcessor(video_path)
        visualizer = Visualizer()
        
        # Set output video writer
        processor.set_output(output_path)
        
        stats = {
            "total_frames": processor.total_frames,
            "detections": 0,
            "signs_detected": {}
        }
        
        try:
            for frame_num, frame in enumerate(processor):
                # Detect signs in frame
                results = self.detect_frame(frame)
                
                if results and results.boxes:
                    # Annotate frame
                    frame = self._annotate_frame(
                        frame,
                        results,
                        visualizer
                    )
                    stats["detections"] += len(results.boxes)
                    
                    # Track sign types
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        sign_code = self.class_mapper.id_to_belarusian(cls_id)
                        if sign_code:
                            stats["signs_detected"][sign_code] = \
                                stats["signs_detected"].get(sign_code, 0) + 1
                
                # Write annotated frame
                processor.write_frame(frame)
                
                # Progress logging
                if (frame_num + 1) % 30 == 0:
                    progress = ((frame_num + 1) / processor.total_frames) * 100
                    logger.info(f"Progress: {frame_num + 1}/{processor.total_frames} ({progress:.1f}%)")
            
            processor.close()
            logger.info(f"Video saved to: {output_path}")
            logger.info(f"Total detections: {stats['detections']}")
            
        except Exception as e:
            processor.close()
            logger.error(f"Video processing failed: {e}")
            raise
        
        return stats
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        results: Results,
        visualizer: Visualizer
    ) -> np.ndarray:
        """Annotate frame with detections.
        
        Args:
            frame: Input frame
            results: YOLO detection results
            visualizer: Visualizer instance
            
        Returns:
            Annotated frame
        """
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            sign_code = self.class_mapper.id_to_belarusian(cls_id)
            if not sign_code:
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Create label
            label = f"{sign_code} {conf:.0%}"
            
            # Draw on frame
            frame = visualizer.draw_detection(
                frame,
                (x1, y1, x2, y2),
                label,
                cls_id
            )
        
        return frame
    
    def get_info(self) -> Dict[str, Any]:
        """Get detector information.
        
        Returns:
            Dictionary with detector info
        """
        return {
            "model_info": self.model.get_model_info(),
            "conf_threshold": self.conf_threshold,
            "num_classes": self.class_mapper.get_num_classes()
        }
