"""Visualization utilities for detection results."""

from typing import Tuple, Optional
import numpy as np
import cv2


class Visualizer:
    """Visualize detection results on frames.
    
    Draws bounding boxes, labels, and confidence scores on frames.
    """
    
    def __init__(self, num_classes: int = 155) -> None:
        """Initialize visualizer.
        
        Args:
            num_classes: Number of classes for color generation
        """
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(num_classes, 3))
    
    def draw_detection(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        class_id: int,
        alpha: float = 0.6,
        thickness: int = 2
    ) -> np.ndarray:
        """Draw detection on frame.
        
        Args:
            frame: Input frame (BGR)
            bbox: Bounding box (x1, y1, x2, y2)
            label: Detection label (e.g., "2.1 95%")
            class_id: Class ID for color selection
            alpha: Transparency for background box
            thickness: Line thickness
            
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = bbox
        color = tuple(map(int, self.colors[class_id]))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background and text
        frame = self._draw_label(
            frame,
            label,
            (x1, y1),
            color,
            alpha
        )
        
        return frame
    
    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float = 0.6
    ) -> np.ndarray:
        """Draw label with background on frame.
        
        Args:
            frame: Input frame
            label: Label text
            position: Top-left position (x, y)
            color: Background color (BGR)
            alpha: Background transparency
            
        Returns:
            Frame with label drawn
        """
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_color = (255, 255, 255)  # White
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Calculate label background position
        padding = 5
        bg_x1 = x
        bg_y1 = y - text_h - 2 * padding
        bg_x2 = x + text_w + 2 * padding
        bg_y2 = y
        
        # Handle edge cases (label off-screen)
        if bg_y1 < 0:
            bg_y1 = y
            bg_y2 = y + text_h + 2 * padding
            text_y = y + text_h + padding
        else:
            text_y = y - padding
        
        # Draw semi-transparent background
        frame = self._draw_transparent_box(
            frame,
            bg_x1, bg_y1,
            bg_x2, bg_y2,
            color,
            alpha
        )
        
        # Draw text
        cv2.putText(
            frame, label,
            (x + padding, text_y),
            font, font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        
        return frame
    
    @staticmethod
    def _draw_transparent_box(
        frame: np.ndarray,
        x1: int, y1: int,
        x2: int, y2: int,
        color: Tuple[int, int, int],
        alpha: float = 0.6
    ) -> np.ndarray:
        """Draw semi-transparent rectangle on frame.
        
        Args:
            frame: Input frame
            x1, y1: Top-left coordinates
            x2, y2: Bottom-right coordinates
            color: Fill color (BGR)
            alpha: Transparency (0-1)
            
        Returns:
            Frame with rectangle drawn
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
