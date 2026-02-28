#!/usr/bin/env python3
"""
Data Source Analysis Script

Analyzes the structure, quality, and statistics of all raw data sources
(RTSD, calendar photos, partner datasets) to inform the data consolidation
and rebuilding pipeline.

Usage:
    python scripts/analyze_data_sources.py
    python scripts/analyze_data_sources.py --source rtsd
    python scripts/analyze_data_sources.py --detailed
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import argparse
from PIL import Image
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClassStats:
    """Statistics for a single class."""
    class_id: int
    class_name: str
    count: int
    min_bbox_area: float = float('inf')
    max_bbox_area: float = 0
    avg_bbox_area: float = 0
    images_with_class: int = 0


@dataclass
class DatasetStats:
    """Overall dataset statistics."""
    name: str
    image_count: int = 0
    annotation_count: int = 0
    class_count: int = 0
    class_stats: Dict[int, ClassStats] = field(default_factory=dict)
    image_sizes: Dict[str, int] = field(default_factory=lambda: {
        'min_width': float('inf'),
        'max_width': 0,
        'min_height': float('inf'),
        'max_height': 0,
    })
    bbox_sizes: Dict[str, float] = field(default_factory=lambda: {
        'min_area': float('inf'),
        'max_area': 0,
        'avg_area': 0,
    })
    missing_files: List[str] = field(default_factory=list)
    invalid_annotations: List[Tuple[str, str]] = field(default_factory=list)
    duplicates: List[Tuple[str, str]] = field(default_factory=list)


class RTSDAnalyzer:
    """Analyzes RTSD dataset from COCO annotations."""

    def __init__(self, rtsd_path: Path = Path('data/raw/rtsd')):
        self.rtsd_path = Path(rtsd_path)
        self.stats = DatasetStats(name='RTSD')

    def analyze(self) -> DatasetStats:
        """Run complete analysis of RTSD dataset."""
        logger.info(f"Analyzing RTSD dataset at {self.rtsd_path}")

        # Check required files
        train_anno_path = self.rtsd_path / 'train_anno.json'
        val_anno_path = self.rtsd_path / 'val_anno.json'
        frames_path = self.rtsd_path / 'rtsd-frames'

        if not train_anno_path.exists():
            logger.error(f"Train annotations not found: {train_anno_path}")
            return self.stats

        # Load training annotations
        logger.info("Loading training annotations...")
        with open(train_anno_path) as f:
            train_data = json.load(f)

        # Load validation annotations if available
        val_data = None
        if val_anno_path.exists():
            logger.info("Loading validation annotations...")
            with open(val_anno_path) as f:
                val_data = json.load(f)

        # Process training data
        self._process_coco_data(train_data, frames_path, 'train')

        # Process validation data if available
        if val_data:
            self._process_coco_data(val_data, frames_path, 'val')

        self._compute_aggregate_stats()
        return self.stats

    def _process_coco_data(self, data: Dict, frames_path: Path, split: str) -> None:
        """Process COCO format dataset."""
        images = {img['id']: img for img in data.get('images', [])}
        annotations = data.get('annotations', [])
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}

        self.stats.image_count += len(images)
        self.stats.class_count = len(categories)

        # Track which images have which classes
        images_per_class = defaultdict(set)
        bbox_areas_per_class = defaultdict(list)

        logger.info(f"Processing {len(annotations)} {split} annotations...")

        for ann_idx, ann in enumerate(annotations):
            if ann_idx % 10000 == 0:
                logger.debug(f"Processing annotation {ann_idx}/{len(annotations)}")

            self.stats.annotation_count += 1

            image_id = ann.get('image_id')
            category_id = ann.get('category_id')
            bbox = ann.get('bbox', [])

            # Validate annotation
            if not bbox or len(bbox) != 4:
                img_name = images.get(image_id, {}).get('file_name', 'unknown')
                self.stats.invalid_annotations.append(
                    (img_name, f"Invalid bbox: {bbox}")
                )
                continue

            x, y, w, h = bbox
            bbox_area = w * h

            # Update class statistics
            if category_id not in self.stats.class_stats:
                self.stats.class_stats[category_id] = ClassStats(
                    class_id=category_id,
                    class_name=categories.get(category_id, f'class_{category_id}'),
                    count=0,
                )

            class_stat = self.stats.class_stats[category_id]
            class_stat.count += 1
            class_stat.min_bbox_area = min(class_stat.min_bbox_area, bbox_area)
            class_stat.max_bbox_area = max(class_stat.max_bbox_area, bbox_area)

            images_per_class[category_id].add(image_id)
            bbox_areas_per_class[category_id].append(bbox_area)

            # Update bbox size statistics
            self.stats.bbox_sizes['min_area'] = min(
                self.stats.bbox_sizes['min_area'], bbox_area
            )
            self.stats.bbox_sizes['max_area'] = max(
                self.stats.bbox_sizes['max_area'], bbox_area
            )

        # Update image counts per class and compute averages
        for class_id, bbox_areas in bbox_areas_per_class.items():
            self.stats.class_stats[class_id].images_with_class = len(
                images_per_class[class_id]
            )
            self.stats.class_stats[class_id].avg_bbox_area = sum(bbox_areas) / len(
                bbox_areas
            )

        # Check for missing files
        logger.info("Verifying image files exist...")
        for image_id, image_info in images.items():
            file_name = image_info['file_name']
            # Handle both relative paths (with rtsd-frames/ prefix) and direct paths
            if file_name.startswith('rtsd-frames/'):
                file_path = frames_path / file_name.replace('rtsd-frames/', '')
            else:
                file_path = frames_path / file_name
            
            if not file_path.exists():
                self.stats.missing_files.append(image_info['file_name'])
            else:
                # Update image size statistics
                try:
                    img = Image.open(file_path)
                    width, height = img.size
                    self.stats.image_sizes['min_width'] = min(
                        self.stats.image_sizes['min_width'], width
                    )
                    self.stats.image_sizes['max_width'] = max(
                        self.stats.image_sizes['max_width'], width
                    )
                    self.stats.image_sizes['min_height'] = min(
                        self.stats.image_sizes['min_height'], height
                    )
                    self.stats.image_sizes['max_height'] = max(
                        self.stats.image_sizes['max_height'], height
                    )
                except Exception as e:
                    self.stats.invalid_annotations.append(
                        (image_info['file_name'], f"Cannot open image: {e}")
                    )

    def _compute_aggregate_stats(self) -> None:
        """Compute aggregate statistics."""
        if self.stats.annotation_count > 0:
            all_areas = []
            for class_stat in self.stats.class_stats.values():
                if class_stat.avg_bbox_area > 0:
                    all_areas.append(class_stat.avg_bbox_area)

            if all_areas:
                self.stats.bbox_sizes['avg_area'] = sum(all_areas) / len(all_areas)


class DataSourcesAnalyzer:
    """Main analyzer for all data sources."""

    def __init__(self, detailed: bool = False):
        self.detailed = detailed

    def analyze_all(self) -> None:
        """Analyze all available data sources."""
        logger.info("=" * 80)
        logger.info("DATA SOURCE ANALYSIS")
        logger.info("=" * 80)

        # Analyze RTSD
        rtsd_analyzer = RTSDAnalyzer()
        rtsd_stats = rtsd_analyzer.analyze()
        self._print_stats(rtsd_stats)

        # Check for other sources
        self._check_calendar_photos()
        self._check_partner_datasets()

        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)

    def _print_stats(self, stats: DatasetStats) -> None:
        """Pretty print dataset statistics."""
        print("\n" + "=" * 80)
        print(f"DATASET: {stats.name}")
        print("=" * 80)

        print(f"\n📊 OVERVIEW:")
        print(f"  • Total images: {stats.image_count:,}")
        print(f"  • Total annotations: {stats.annotation_count:,}")
        print(f"  • Total classes: {stats.class_count}")
        print(f"  • Avg annotations per image: {stats.annotation_count / max(stats.image_count, 1):.2f}")

        if stats.image_sizes['min_width'] != float('inf'):
            print(f"\n📐 IMAGE SIZES:")
            print(f"  • Width:  {stats.image_sizes['min_width']} - {stats.image_sizes['max_width']} px")
            print(f"  • Height: {stats.image_sizes['min_height']} - {stats.image_sizes['max_height']} px")

        print(f"\n📦 BOUNDING BOX STATISTICS:")
        print(f"  • Min area: {stats.bbox_sizes['min_area']:.0f} px²")
        print(f"  • Max area: {stats.bbox_sizes['max_area']:.0f} px²")
        print(f"  • Avg area: {stats.bbox_sizes['avg_area']:.0f} px²")

        if self.detailed and stats.class_stats:
            print(f"\n📋 CLASS DISTRIBUTION (top 15):")
            sorted_classes = sorted(
                stats.class_stats.values(),
                key=lambda x: x.count,
                reverse=True
            )
            for i, class_stat in enumerate(sorted_classes[:15], 1):
                print(f"  {i:2d}. {class_stat.class_name:15s} "
                      f"count={class_stat.count:5d} "
                      f"images={class_stat.images_with_class:5d} "
                      f"avg_area={class_stat.avg_bbox_area:8.0f} px²")

            if len(sorted_classes) > 15:
                print(f"  ... and {len(sorted_classes) - 15} more classes")

        if stats.missing_files:
            print(f"\n⚠️  MISSING FILES: {len(stats.missing_files)}")
            for file_path in stats.missing_files[:5]:
                print(f"  • {file_path}")
            if len(stats.missing_files) > 5:
                print(f"  ... and {len(stats.missing_files) - 5} more")

        if stats.invalid_annotations:
            print(f"\n❌ INVALID ANNOTATIONS: {len(stats.invalid_annotations)}")
            for file_path, reason in stats.invalid_annotations[:5]:
                print(f"  • {file_path}: {reason}")
            if len(stats.invalid_annotations) > 5:
                print(f"  ... and {len(stats.invalid_annotations) - 5} more")

    def _check_calendar_photos(self) -> None:
        """Check if calendar photos directory exists."""
        calendar_path = Path('datasets/raw_sources/calendar_photos')
        if calendar_path.exists():
            print("\n✅ Calendar photos directory found")
            raw_path = calendar_path / 'raw'
            if raw_path.exists():
                image_files = list(raw_path.glob('*.jpg')) + list(raw_path.glob('*.png'))
                print(f"   • Found {len(image_files)} image files")
        else:
            print("\n❌ Calendar photos directory not found (expected)")

    def _check_partner_datasets(self) -> None:
        """Check if partner datasets directory exists."""
        partner_path = Path('datasets/raw_sources/partner_datasets')
        if partner_path.exists():
            print("\n✅ Partner datasets directory found")
            subdirs = [d for d in partner_path.iterdir() if d.is_dir()]
            if subdirs:
                print(f"   • Found {len(subdirs)} partner dataset(s):")
                for subdir in subdirs:
                    print(f"     - {subdir.name}")
        else:
            print("\n❌ Partner datasets directory not found (expected)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze raw data sources for traffic sign detection pipeline'
    )
    parser.add_argument(
        '--source',
        choices=['rtsd', 'calendar', 'partner', 'all'],
        default='all',
        help='Data source to analyze'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed class distribution statistics'
    )

    args = parser.parse_args()

    if args.source == 'all' or args.source == 'rtsd':
        analyzer = DataSourcesAnalyzer(detailed=args.detailed)
        analyzer.analyze_all()
    elif args.source == 'calendar':
        print("Calendar photos analysis not yet implemented")
    elif args.source == 'partner':
        print("Partner datasets analysis not yet implemented")


if __name__ == '__main__':
    main()
