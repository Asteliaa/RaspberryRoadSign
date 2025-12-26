#!/usr/bin/env python3
"""
RTSD to YOLO Converter (80/10/10 Split) - Clean Version
Minimal logging, no excessive prints
"""

import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Minimal logging - only errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10
DEFAULT_RANDOM_SEED = 42
MIN_SAMPLES_PER_CLASS = 5

CLASS_NAMES = [
    "2_1", "1_23", "1_17", "3_24", "8_2_1", "5_20", "5_19_1", "5_16",
    "3_25", "6_16", "7_15", "2_2", "2_4", "8_13_1", "4_2_1", "1_20_3",
    "1_25", "3_4", "8_3_2", "3_4_1", "4_1_6", "4_2_3", "4_1_1", "1_33",
    "5_15_5", "3_27", "1_15", "4_1_2_1", "6_3_1", "8_1_1", "6_7", "5_15_3",
    "7_3", "1_19", "6_4", "8_1_4", "8_8", "1_16", "1_11_1", "6_6",
    "5_15_1", "7_2", "5_15_2", "7_12", "3_18", "5_6", "5_5", "7_4",
    "4_1_2", "8_2_2", "7_11", "1_22", "1_27", "2_3_2", "5_15_2_2", "1_8",
    "3_13", "2_3", "8_3_3", "2_3_3", "7_7", "1_11", "8_13", "1_12_2",
    "1_20", "1_12", "3_32", "2_5", "3_1", "4_8_2", "3_20", "3_2",
    "2_3_6", "5_22", "5_18", "2_3_5", "7_5", "8_4_1", "3_14", "1_2",
    "1_20_2", "4_1_4", "7_6", "8_1_3", "8_3_1", "4_3", "4_1_5", "8_2_3",
    "8_2_4", "1_31", "3_10", "4_2_2", "7_1", "3_28", "4_1_3", "5_4",
    "5_3", "6_8_2", "3_31", "6_2", "1_21", "3_21", "1_13", "1_14",
    "2_3_4", "4_8_3", "6_15_2", "2_6", "3_18_2", "4_1_2_2", "1_7", "3_19",
    "1_18", "2_7", "8_5_4", "5_15_7", "5_14", "5_21", "1_1", "6_15_1",
    "8_6_4", "8_15", "4_5", "3_11", "8_18", "8_4_4", "3_30", "5_7_1",
    "5_7_2", "1_5", "3_29", "6_15_3", "5_12", "3_16", "1_30", "5_11",
    "1_6", "8_6_2", "6_8_3", "3_12", "3_33", "8_4_3", "5_8", "8_14",
    "8_17", "3_6", "1_26", "8_5_2", "6_8_1", "5_17", "1_10", "8_16",
    "7_18", "7_14", "8_23"
]


def find_rtsd_structure() -> Dict[str, Path]:
    """Locate RTSD dataset"""
    cwd = Path.cwd()
    rtsd_candidates = [
        cwd / "data" / "raw" / "rtsd",
        cwd / "datasets" / "rtsd_raw",
        cwd / "rtsd",
    ]

    rtsd_root = None
    for candidate_path in rtsd_candidates:
        if candidate_path.exists():
            rtsd_root = candidate_path
            break

    if not rtsd_root:
        raise FileNotFoundError(
            "RTSD dataset not found in: "
            "data/raw/rtsd/, datasets/rtsd_raw/, rtsd/"
        )

    anno_files = {
        "train": rtsd_root / "train_anno.json",
        "val": rtsd_root / "val_anno.json",
        "anno": rtsd_root / "anno.json",
        "train_reduced": rtsd_root / "train_anno_reduced.json",
    }

    found_annos = {k: v for k, v in anno_files.items() if v.exists()}

    images_dir = None
    for candidate in [rtsd_root / "rtsd-frames", rtsd_root / "images", rtsd_root / "frames"]:
        if candidate.exists():
            images_dir = candidate
            break

    if not images_dir:
        raise FileNotFoundError(f"Images directory not found in {rtsd_root}")

    output_base_dir = cwd / "datasets" / "rtsd_yolo"

    return {
        "rtsd_root": rtsd_root,
        "annotations": found_annos, # type: ignore
        "images_dir": images_dir,
        "output_base_dir": output_base_dir,
    }


def select_annotation_file(anno_dict: Dict[str, Path]) -> Path:
    """Select best annotation file by priority"""
    priority = ["train", "train_reduced", "anno", "val"]
    for key in priority:
        if key in anno_dict:
            return anno_dict[key]
    raise ValueError("No suitable annotation files found")


def load_coco_data(json_file: Path) -> Tuple[Dict[int, str], List[Dict]]:
    """Load COCO format annotations"""
    with open(json_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    image_id_to_name = {img["id"]: img["file_name"]
                        for img in coco_data.get("images", [])}
    annotations = coco_data.get("annotations", [])

    return image_id_to_name, annotations


def bbox_to_yolo(
    x: float,
    y: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """Convert COCO bbox to YOLO normalized format"""
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height

    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_width = max(0, min(1, norm_width))
    norm_height = max(0, min(1, norm_height))

    return x_center, y_center, norm_width, norm_height


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    return img.shape[0], img.shape[1]


def create_yolo_labels_and_load_dataset(
    annotations: List[Dict[str, Any]],
    image_id_to_name: Dict[int, str],
    images_dir: Path,
    labels_dir: Path,
) -> Tuple[Counter, List[Dict[str, Any]]]:
    """Create YOLO labels and load dataset"""
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_distribution = Counter()
    data_with_classes = []

    no_image_id = 0
    image_not_found = 0
    bbox_invalid = 0
    success_count = 0

    for ann in tqdm(annotations, desc="Processing annotations", total=len(annotations)):
        try:
            image_id = ann.get("image_id")
            if image_id is None:
                no_image_id += 1
                continue

            image_name = image_id_to_name.get(image_id)
            if not image_name:
                image_not_found += 1
                continue

            category_id = ann.get("category_id")
            if category_id is None:
                continue

            class_id = category_id - 1

            if not (0 <= class_id < len(CLASS_NAMES)):
                continue

            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                bbox_invalid += 1
                continue

            x, y, bbox_width, bbox_height = bbox[:4]

            # Find image file
            image_path = None
            test_path = images_dir / image_name
            if test_path.exists():
                image_path = test_path
            else:
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                    alt_path = images_dir / f"{Path(image_name).stem}{ext}"
                    if alt_path.exists():
                        image_path = alt_path
                        break

                if not image_path:
                    matching = list(images_dir.glob(
                        f"**/{Path(image_name).name}"))
                    if matching:
                        image_path = matching[0]

            if not image_path or not image_path.exists():
                image_not_found += 1
                continue

            img_height, img_width = get_image_dimensions(image_path)

            center_x, center_y, norm_width, norm_height = bbox_to_yolo(
                float(x), float(y), float(bbox_width), float(bbox_height),
                img_width, img_height,
            )

            label_file = labels_dir / f"{image_path.stem}.txt"
            with open(label_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

            data_with_classes.append({
                "image": image_path,
                "label": label_file,
                "class_id": class_id,
            })

            class_distribution[class_id] += 1
            success_count += 1

        except (IOError, ValueError, IndexError, TypeError):
            continue

    if success_count == 0:
        raise ValueError("No valid samples processed")

    return class_distribution, data_with_classes


def copy_split_files(
    data_list: List[Dict[str, Any]], split_name: str, output_base_dir: Path
) -> None:
    """Copy image and label files to split directories"""
    img_dir = output_base_dir / "images" / split_name
    lbl_dir = output_base_dir / "labels" / split_name

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(data_list, desc=f"Copying {split_name}", leave=False):
        shutil.copy(str(item["image"]), str(img_dir / item["image"].name))
        shutil.copy(str(item["label"]), str(lbl_dir / item["label"].name))


def verify_split_integrity(output_base_dir: Path) -> bool:
    """Verify dataset split integrity"""
    all_valid = True
    for split in ["train", "val", "test"]:
        img_dir = output_base_dir / "images" / split
        lbl_dir = output_base_dir / "labels" / split

        if not img_dir.exists():
            continue

        img_count = len(list(img_dir.glob("*")))
        lbl_count = len(list(lbl_dir.glob("*.txt")))

        if img_count != lbl_count:
            all_valid = False

    return all_valid


def save_data_yaml(class_names: Dict[int, str], output_base_dir: Path) -> None:
    """Save data.yaml for YOLOv11"""
    data_yaml = {
        "path": str(output_base_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output_base_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)


def save_split_statistics(
    data_with_classes: List[Dict[str, Any]],
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    class_distribution: Counter,
    output_base_dir: Path,
) -> None:
    """Save split statistics to JSON"""
    total_samples = len(data_with_classes)

    split_stats = {
        "total_samples": total_samples,
        "train": {
            "count": len(train_data),
            "percentage": len(train_data) / total_samples * 100,
        },
        "val": {
            "count": len(val_data),
            "percentage": len(val_data) / total_samples * 100,
        },
        "test": {
            "count": len(test_data),
            "percentage": len(test_data) / total_samples * 100,
        },
        "class_distribution": {str(k): v for k, v in class_distribution.items()},
        "num_classes": len(class_distribution),
    }

    stats_file = output_base_dir / "split_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)


def stratified_split(
    anno_file: Path,
    images_dir: Path,
    output_base_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
) -> Dict[str, Any]:
    """Perform stratified train/val/test split"""

    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, "Sum of ratios must equal 1.0"

    np.random.seed(random_seed)

    # Create directories
    for split in ["train", "val", "test"]:
        (output_base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Load and process data
    image_id_to_name, annotations = load_coco_data(anno_file)
    temp_labels_dir = output_base_dir / "labels" / "temp"
    class_distribution, data_with_classes = create_yolo_labels_and_load_dataset(
        annotations, image_id_to_name, images_dir, temp_labels_dir
    )

    print(f"\n✓ Loaded {len(data_with_classes)} samples")

    # Prepare for split
    stratify_labels = [item["class_id"] for item in data_with_classes]
    class_counts = Counter(stratify_labels)
    min_count = min(class_counts.values())

    # Perform split (stratified if possible)
    if min_count >= 2:
        train_data, temp_data = train_test_split(
            data_with_classes,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            stratify=stratify_labels,
        )

        temp_stratify_labels = [item["class_id"] for item in temp_data]
        val_test_split = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_test_split),
            random_state=random_seed,
            stratify=temp_stratify_labels,
        )
    else:
        train_data, temp_data = train_test_split(
            data_with_classes,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
        )

        val_test_split = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_test_split),
            random_state=random_seed,
        )

    # Copy files to splits
    copy_split_files(train_data, "train", output_base_dir)
    copy_split_files(val_data, "val", output_base_dir)
    copy_split_files(test_data, "test", output_base_dir)

    # Save statistics and config
    save_split_statistics(
        data_with_classes, train_data, val_data, test_data,
        class_distribution, output_base_dir
    )

    class_names = {i: name for i, name in enumerate(CLASS_NAMES)}
    save_data_yaml(class_names, output_base_dir)

    # Cleanup
    if temp_labels_dir.exists():
        shutil.rmtree(temp_labels_dir)

    total_samples = len(data_with_classes)
    return {
        "total_samples": total_samples,
        "train": {"count": len(train_data), "percentage": len(train_data) / total_samples * 100},
        "val": {"count": len(val_data), "percentage": len(val_data) / total_samples * 100},
        "test": {"count": len(test_data), "percentage": len(test_data) / total_samples * 100},
        "classes": len(class_distribution),
    }


def main() -> None:
    """Main entry point"""
    try:
        print("RTSD to YOLO Converter")

        paths = find_rtsd_structure()
        anno_file = select_annotation_file(paths["annotations"]) # type: ignore
        print(f"✓ Using: {anno_file.name}")

        result = stratified_split(
            anno_file=anno_file,
            images_dir=paths["images_dir"],
            output_base_dir=paths["output_base_dir"],
            min_samples_per_class=MIN_SAMPLES_PER_CLASS,
        )

        print(
            f"\nTrain:  {result['train']['count']:6d} ({result['train']['percentage']:5.1f}%)")
        print(
            f"Val:    {result['val']['count']:6d} ({result['val']['percentage']:5.1f}%)")
        print(
            f"Test:   {result['test']['count']:6d} ({result['test']['percentage']:5.1f}%)")
        print(f"Total:  {result['total_samples']:6d}")
        print(f"Classes: {result['classes']}")
        print(
            f"\nOutput: {paths['output_base_dir'].relative_to(Path.cwd())}/")
        print("Ready: python train_yolo_gpu.py")

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user\n")

    except Exception as error:
        print(f"\n✗ Error: {error}\n")
        raise


if __name__ == "__main__":
    main()
