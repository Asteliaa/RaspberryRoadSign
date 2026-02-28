"""
Data source adapters for different dataset formats.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

from ..base import BaseAdapter, Detection

logger = logging.getLogger(__name__)


class RTSDAdapter(BaseAdapter):
    """Adapter for RTSD dataset with COCO format annotations."""

    def __init__(self, source_path: Path = Path('data/raw/rtsd')):
        """Initialize RTSD adapter.
        
        Args:
            source_path: Path to RTSD directory containing:
                - train_anno.json (training annotations)
                - val_anno.json (validation annotations, optional)
                - rtsd-frames/ (image directory)
        """
        super().__init__(source_path)
        self.frames_path = self.source_path / 'rtsd-frames'
        self.train_anno_path = self.source_path / 'train_anno.json'
        self.val_anno_path = self.source_path / 'val_anno.json'
        
        self._detections: List[Detection] = []
        self._is_loaded = False

    def load_detections(self) -> List[Detection]:
        """Load all detections from RTSD annotations.
        
        Returns:
            List of Detection objects from training and validation sets
        """
        if self._is_loaded:
            return self._detections

        logger.info(f"Loading RTSD detections from {self.source_path}")

        # Load training annotations
        if self.train_anno_path.exists():
            logger.info("Loading training annotations...")
            self._load_coco_file(self.train_anno_path)
        else:
            logger.warning(f"Train annotations not found: {self.train_anno_path}")

        # Load validation annotations if available
        if self.val_anno_path.exists():
            logger.info("Loading validation annotations...")
            self._load_coco_file(self.val_anno_path)

        self._is_loaded = True
        logger.info(f"Loaded {len(self._detections)} total detections")
        return self._detections

    def _load_coco_file(self, anno_path: Path) -> None:
        """Load detections from a COCO format JSON file.
        
        Args:
            anno_path: Path to COCO annotations JSON file
        """
        with open(anno_path) as f:
            data = json.load(f)

        # Build lookup tables
        images = {img['id']: img for img in data.get('images', [])}
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        annotations = data.get('annotations', [])

        logger.info(f"Processing {len(annotations)} annotations from {anno_path.name}")

        for ann in tqdm(annotations, desc="Processing annotations"):
            image_id = ann.get('image_id')
            category_id = ann.get('category_id')
            bbox = ann.get('bbox')

            # Validate annotation
            if not bbox or len(bbox) != 4:
                logger.debug(f"Skipping annotation with invalid bbox: {ann}")
                continue

            if image_id not in images:
                logger.debug(f"Image ID not found: {image_id}")
                continue

            image_info = images[image_id]
            
            # Construct image path
            file_name = image_info['file_name']
            if file_name.startswith('rtsd-frames/'):
                image_path = self.frames_path / file_name.replace('rtsd-frames/', '')
            else:
                image_path = self.frames_path / file_name

            # Create detection
            detection = Detection(
                image_id=str(image_id),
                image_path=image_path,
                class_id=category_id,
                class_name=categories.get(category_id, f'class_{category_id}'),
                bbox=tuple(bbox),  # [x, y, width, height]
                is_crowd=ann.get('iscrowd', 0) != 0
            )

            self._detections.append(detection)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate RTSD source data.
        
        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        errors = []

        # Check required files exist
        if not self.train_anno_path.exists():
            errors.append(f"Missing training annotations: {self.train_anno_path}")
        
        if not self.frames_path.exists():
            errors.append(f"Missing frames directory: {self.frames_path}")

        # Check annotations are valid JSON
        try:
            with open(self.train_anno_path) as f:
                data = json.load(f)
                if 'images' not in data or 'annotations' not in data:
                    errors.append("Train annotations missing 'images' or 'annotations' keys")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in train annotations: {e}")

        if self.val_anno_path.exists():
            try:
                with open(self.val_anno_path) as f:
                    data = json.load(f)
                    if 'images' not in data or 'annotations' not in data:
                        errors.append("Val annotations missing 'images' or 'annotations' keys")
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in val annotations: {e}")

        # Check sample image files exist
        self.load_detections()
        if self._detections:
            missing_count = 0
            sample_size = min(100, len(self._detections))
            for detection in self._detections[:sample_size]:
                if not detection.image_path.exists():
                    missing_count += 1

            if missing_count > 0:
                missing_pct = (missing_count / sample_size) * 100
                errors.append(
                    f"Sample check: {missing_count}/{sample_size} image files missing ({missing_pct:.1f}%)"
                )

        is_valid = len(errors) == 0
        return is_valid, errors

    def __len__(self) -> int:
        """Return number of images in RTSD."""
        if not self._is_loaded:
            self.load_detections()
        return len(set(d.image_id for d in self._detections))

    def get_classes(self) -> Dict[int, str]:
        """Get mapping of class IDs to class names.
        
        Returns:
            Dictionary mapping class_id -> class_name
        """
        if not self._is_loaded:
            self.load_detections()
        
        return {
            d.class_id: d.class_name
            for d in self._detections
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of detections per class.
        
        Returns:
            Dictionary mapping class_name -> count
        """
        if not self._is_loaded:
            self.load_detections()

        distribution = {}
        for detection in self._detections:
            key = f"{detection.class_id}_{detection.class_name}"
            distribution[key] = distribution.get(key, 0) + 1

        return distribution
