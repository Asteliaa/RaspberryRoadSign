"""
Base adapter classes for different data source formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a single detection in an image."""
    image_id: str
    image_path: Path
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # [x, y, width, height]
    is_crowd: bool = False


class BaseAdapter(ABC):
    """Base class for data source adapters."""

    def __init__(self, source_path: Path):
        """Initialize adapter with source path."""
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

    @abstractmethod
    def load_detections(self) -> List[Detection]:
        """Load all detections from source.
        
        Returns:
            List of Detection objects
        """
        pass

    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the source data.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass

    def __len__(self) -> int:
        """Return number of images in source."""
        raise NotImplementedError
