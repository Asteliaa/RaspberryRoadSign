"""Video file processing utilities."""

from pathlib import Path
from typing import Union, Iterator, Optional, Tuple
import logging
import cv2
import numpy as np


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process video files frame by frame.
    
    Handles video reading, frame extraction, and writing annotated videos.
    
    Attributes:
        video_path: Path to input video
        fps: Frames per second
        width: Frame width
        height: Frame height
        total_frames: Total number of frames
    """
    
    def __init__(self, video_path: Union[str, Path]) -> None:
        """Initialize video processor.
        
        Args:
            video_path: Path to video file
            
        Raises:
            FileNotFoundError: If video not found
            RuntimeError: If video can't be opened
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_frame = 0
        
        logger.info(
            f"Video loaded: {self.width}x{self.height} @ {self.fps} FPS, "
            f"{self.total_frames} frames"
        )
    
    def set_output(
        self,
        output_path: Union[str, Path],
        codec: str = 'mp4v'
    ) -> None:
        """Set output video writer.
        
        Args:
            output_path: Path to save output video
            codec: Video codec ('mp4v', 'XVID', 'MJPG')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")
        
        logger.info(f"Output video writer initialized: {output_path}")
    
    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video.
        
        Args:
            frame: Frame to write
            
        Raises:
            RuntimeError: If writer not initialized
        """
        if self.writer is None:
            raise RuntimeError("Output video writer not initialized. Call set_output() first.")
        
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames in video.
        
        Yields:
            BGR frame as numpy array
        """
        self.current_frame = 0
        return self
    
    def __next__(self) -> np.ndarray:
        """Get next frame.
        
        Returns:
            BGR frame as numpy array
            
        Raises:
            StopIteration: When video ends
        """
        ret, frame = self.cap.read()
        
        if not ret:
            raise StopIteration
        
        self.current_frame += 1
        return frame
    
    def close(self) -> None:
        """Close video files."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        logger.info("Video resources released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
