"""
Main dataset rebuilder for traffic sign detection pipeline.

Consolidates multiple data sources, generates YOLO labels (6-group format),
extracts CNN training crops (155 classes), and creates train/val/test splits.

Usage:
    python scripts/rebuild_datasets.py
    python scripts/rebuild_datasets.py --output-dir /custom/path
    python scripts/rebuild_datasets.py --train-split 0.7 --val-split 0.15
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse
from dataclasses import dataclass, asdict
import random

import numpy as np
from tqdm import tqdm
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset rebuilding."""
    source_path: Path = Path('data/raw/rtsd')
    output_dir: Path = Path('datasets')
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    min_bbox_size: int = 16  # Minimum bbox dimension in pixels
    max_bbox_size: int = 500  # Maximum bbox dimension in pixels


class YOLOGroupMapper:
    """Maps RTSD classes (155) to YOLO groups (6)."""

    # Mapping from Belarusian sign codes to 6-group categories
    # Based on GOST 23457-85 classification
    CATEGORY_MAPPING = {
        # WARNING signs (1.x) -> Group 0
        'warning': [
            '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7', '1_8', '1_9', '1_10',
            '1_11', '1_12', '1_13', '1_14', '1_15', '1_16', '1_17', '1_18', '1_19',
            '1_20', '1_21', '1_22', '1_23', '1_24', '1_25', '1_26', '1_27', '1_28',
            '1_29', '1_30', '1_31', '1_32'
        ],
        # PRIORITY signs (2.x) -> Group 1
        'priority': [
            '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7'
        ],
        # PROHIBITORY signs (3.x) -> Group 2
        'prohibitory': [
            '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '3_10',
            '3_11', '3_12', '3_13', '3_14', '3_15', '3_16', '3_17', '3_18', '3_19',
            '3_20', '3_21', '3_22', '3_23', '3_24', '3_25', '3_26', '3_27', '3_28',
            '3_29', '3_30', '3_31', '3_32', '3_33', '3_34', '3_35'
        ],
        # MANDATORY signs (4.x) -> Group 3
        'mandatory': [
            '4_1_1', '4_1_2', '4_1_3', '4_1_4', '4_1_5', '4_1_6',
            '4_2', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8'
        ],
        # INFO signs (5.x) -> Group 4
        'info': [
            '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9', '5_10',
            '5_11', '5_12', '5_13', '5_14', '5_15_1', '5_15_2', '5_15_2_1', '5_15_2_2',
            '5_15_3', '5_16', '5_17', '5_18', '5_19_1', '5_19_2', '5_20', '5_21',
            '5_22', '5_23', '5_24', '5_25'
        ],
        # SERVICE/OTHER signs (6.x, 7.x, 8.x, etc.) -> Group 5
        'service': [
            '6_1', '6_2', '6_3', '6_4', '6_5', '6_6', '6_7', '6_8', '6_9', '6_10',
            '6_11', '6_12', '6_13', '6_14', '6_15', '6_16', '6_17', '6_18', '6_19',
            '6_20', '6_21', '6_22', '6_23', '6_24', '6_25', '6_26', '6_27', '6_28',
            '6_29', '6_30', '6_31', '6_32', '6_33', '6_34', '6_35', '6_36',
            '7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7', '7_8', '7_9', '7_10',
            '7_11', '7_12', '7_13', '7_14', '7_15', '7_16',
            '8_1_1', '8_1_2', '8_2_1', '8_2_2', '8_2_3', '8_2_4', '8_2_5', '8_2_6',
            '8_3_1', '8_3_2', '8_3_3'
        ]
    }

    YOLO_GROUP_NAMES = ['warning', 'priority', 'prohibitory', 'mandatory', 'info', 'service']
    YOLO_GROUP_IDS = {name: i for i, name in enumerate(YOLO_GROUP_NAMES)}

    # Build inverse mapping: class_name -> group_id
    CLASS_TO_GROUP = {}
    for group_name, classes in CATEGORY_MAPPING.items():
        group_id = YOLO_GROUP_IDS[group_name]
        for class_name in classes:
            CLASS_TO_GROUP[class_name] = group_id

    @classmethod
    def get_yolo_group(cls, belarusian_code: str) -> int:
        """Get YOLO group ID for a Belarusian traffic sign code.
        
        Args:
            belarusian_code: Sign code like '2_1', '1_23', etc.
            
        Returns:
            YOLO group ID (0-5), or -1 if unknown
        """
        return cls.CLASS_TO_GROUP.get(belarusian_code, -1)

    @classmethod
    def get_all_groups(cls) -> Dict[int, str]:
        """Get all YOLO group mappings.
        
        Returns:
            Dictionary mapping group_id -> group_name
        """
        return dict(enumerate(cls.YOLO_GROUP_NAMES))


class DatasetRebuilder:
    """Main dataset rebuilder orchestrator."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        """Initialize rebuilder with configuration.
        
        Args:
            config: DatasetConfig object, or None to use defaults
        """
        self.config = config or DatasetConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Create output directories
        self.yolo_dir = self.config.output_dir / 'stage1_yolo'
        self.cnn_dir = self.config.output_dir / 'stage2_cnn'

        logger.info(f"Dataset rebuilder initialized")
        logger.info(f"  Source: {self.config.source_path}")
        logger.info(f"  Output: {self.config.output_dir}")
        logger.info(f"  Splits: train={self.config.train_split}, val={self.config.val_split}, test={self.config.test_split}")

    def rebuild(self) -> None:
        """Execute complete dataset rebuilding pipeline."""
        logger.info("Starting dataset rebuild...")

        # Load data
        logger.info("Loading data from RTSD...")
        import sys
        from pathlib import Path as PathlibPath
        sys.path.insert(0, str(PathlibPath(__file__).parent.parent / 'src'))
        from RaspberryRoadSign.data_prep.adapters import RTSDAdapter
        adapter = RTSDAdapter(self.config.source_path)
        detections = adapter.load_detections()
        logger.info(f"Loaded {len(detections)} detections from {len(set(d.image_id for d in detections))} images")

        # Filter and validate detections
        detections = self._filter_detections(detections)
        logger.info(f"After filtering: {len(detections)} detections")

        # Group detections by image
        images_detections = self._group_by_image(detections)
        images = list(images_detections.keys())
        logger.info(f"Processing {len(images)} unique images")

        # Create train/val/test split
        train_images, val_images, test_images = self._split_images(images)
        logger.info(f"Split: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

        # Generate YOLO Stage 1 dataset
        logger.info("Generating YOLO Stage 1 dataset (6-group detection)...")
        self._generate_yolo_dataset(images_detections, train_images, val_images, test_images)

        # Generate CNN Stage 2 dataset
        logger.info("Generating CNN Stage 2 dataset (155-class classification)...")
        self._generate_cnn_dataset(images_detections, train_images, val_images, test_images)

        logger.info("✓ Dataset rebuild complete!")

    def _filter_detections(self, detections) -> List:
        """Filter detections by size constraints.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        skipped = 0

        for det in detections:
            x, y, w, h = det.bbox
            # Skip if bbox is too small or too large
            if min(w, h) < self.config.min_bbox_size or max(w, h) > self.config.max_bbox_size:
                skipped += 1
                continue
            # Skip if image file doesn't exist
            if not det.image_path.exists():
                skipped += 1
                continue
            filtered.append(det)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} detections due to size constraints or missing files")

        return filtered

    def _group_by_image(self, detections) -> Dict[str, List]:
        """Group detections by image ID.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary mapping image_id -> list of detections
        """
        grouped = defaultdict(list)
        for det in detections:
            grouped[det.image_id].append(det)
        return dict(grouped)

    def _split_images(self, images: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Split images into train/val/test sets.
        
        Args:
            images: List of image IDs
            
        Returns:
            Tuple of (train_images, val_images, test_images)
        """
        random.shuffle(images)
        
        n_train = int(len(images) * self.config.train_split)
        n_val = int(len(images) * self.config.val_split)
        
        train = images[:n_train]
        val = images[n_train:n_train + n_val]
        test = images[n_train + n_val:]
        
        return train, val, test

    def _generate_yolo_dataset(self, images_detections: Dict, train_images: List,
                               val_images: List, test_images: List) -> None:
        """Generate YOLO Stage 1 dataset (6-group detection).
        
        Args:
            images_detections: Dictionary mapping image_id -> detections
            train_images: List of training image IDs
            val_images: List of validation image IDs
            test_images: List of test image IDs
        """
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Copy images and generate YOLO labels
        split_info = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        class_counts = defaultdict(int)
        image_counts = {'train': 0, 'val': 0, 'test': 0}

        for split, image_ids in split_info.items():
            logger.info(f"Processing {split} split ({len(image_ids)} images)...")
            
            for image_id in tqdm(image_ids, desc=f"{split} images"):
                detections = images_detections.get(image_id, [])
                if not detections:
                    continue

                source_image_path = detections[0].image_path
                if not source_image_path.exists():
                    continue

                # Copy image
                dest_image_path = self.yolo_dir / split / 'images' / source_image_path.name
                shutil.copy2(source_image_path, dest_image_path)

                # Generate YOLO label file
                label_file = self.yolo_dir / split / 'labels' / (source_image_path.stem + '.txt')
                self._write_yolo_labels(detections, label_file, source_image_path)

                image_counts[split] += 1
                for det in detections:
                    group_id = YOLOGroupMapper.get_yolo_group(det.class_name)
                    if group_id >= 0:
                        class_counts[group_id] += 1

        # Generate data.yaml
        self._generate_data_yaml(class_counts, image_counts)

        logger.info(f"✓ YOLO dataset generated: {self.yolo_dir}")

    def _write_yolo_labels(self, detections, label_file: Path, image_path: Path) -> None:
        """Write YOLO format label file.
        
        Args:
            detections: List of detections for image
            label_file: Path to output label file
            image_path: Path to image (to get dimensions)
        """
        lines = []
        
        # Get image dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            logger.warning(f"Cannot open image {image_path}: {e}")
            return

        for det in detections:
            group_id = YOLOGroupMapper.get_yolo_group(det.class_name)
            if group_id < 0:
                logger.debug(f"Unknown class: {det.class_name}")
                continue

            x, y, w, h = det.bbox
            
            # Convert to YOLO format: center coordinates and dimensions normalized 0-1
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height

            # Clamp to valid range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))

            lines.append(f"{group_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

        if lines:
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines))

    def _generate_data_yaml(self, class_counts: Dict, image_counts: Dict) -> None:
        """Generate data.yaml for YOLO training.
        
        Args:
            class_counts: Dictionary of class detection counts
            image_counts: Dictionary of image counts per split
        """
        yaml_path = self.yolo_dir / 'data.yaml'
        
        groups = YOLOGroupMapper.get_all_groups()
        class_names = {k: v for k, v in groups.items()}

        data = {
            'path': str(self.yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(groups),
            'names': class_names
        }

        with open(yaml_path, 'w') as f:
            # Write as YAML
            f.write(f"path: {data['path']}\n")
            f.write(f"train: {data['train']}\n")
            f.write(f"val: {data['val']}\n")
            f.write(f"test: {data['test']}\n")
            f.write(f"nc: {data['nc']}\n")
            f.write("names:\n")
            for class_id, class_name in sorted(class_names.items()):
                f.write(f"  {class_id}: {class_name}\n")

        logger.info(f"  Generated data.yaml: {yaml_path}")

    def _generate_cnn_dataset(self, images_detections: Dict, train_images: List,
                              val_images: List, test_images: List) -> None:
        """Generate CNN Stage 2 dataset (155-class classification).
        
        Args:
            images_detections: Dictionary mapping image_id -> detections
            train_images: List of training image IDs
            val_images: List of validation image IDs
            test_images: List of test image IDs
        """
        logger.info("CNN dataset generation - placeholder for Phase 4")
        # TODO: Implement in next phase


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Rebuild traffic sign datasets from RTSD source'
    )
    parser.add_argument(
        '--source',
        type=Path,
        default=Path('data/raw/rtsd'),
        help='Path to RTSD source directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('datasets'),
        help='Output directory for rebuilt datasets'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training set fraction (default: 0.8)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation set fraction (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    config = DatasetConfig(
        source_path=args.source,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=1.0 - args.train_split - args.val_split,
        random_seed=args.seed
    )

    rebuilder = DatasetRebuilder(config)
    rebuilder.rebuild()


if __name__ == '__main__':
    main()
