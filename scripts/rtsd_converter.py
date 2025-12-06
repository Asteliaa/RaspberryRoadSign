"""
RTSD to YOLO converter with correct class name mapping.

This converter:
- Loads COCO format annotations from RTSD dataset
- Converts bounding boxes to YOLO format
- Splits dataset into train/val/test with stratification
- Applies correct GOST class names
- Generates data.yaml with proper class names

Expected structure:
- data/raw/rtsd/
  ├── train_anno.json
  ├── val_anno.json
  ├── rtsd-frames/
  └── label_map.json

Output:
- datasets/rtsd_yolo/
  ├── images/train, images/val, images/test
  ├── labels/train, labels/val, labels/test
  ├── data.yaml
  └── split_statistics.json
"""

import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_TRAIN_RATIO = 0.65
DEFAULT_VAL_RATIO = 0.20
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42
MIN_SAMPLES_PER_CLASS = 5

LABEL_MAP = {
    "2_1": 1,
    "1_23": 2,
    "1_17": 3,
    "3_24": 4,
    "8_2_1": 5,
    "5_20": 6,
    "5_19_1": 7,
    "5_16": 8,
    "3_25": 9,
    "6_16": 10,
    "7_15": 11,
    "2_2": 12,
    "2_4": 13,
    "8_13_1": 14,
    "4_2_1": 15,
    "1_20_3": 16,
    "1_25": 17,
    "3_4": 18,
    "8_3_2": 19,
    "3_4_1": 20,
    "4_1_6": 21,
    "4_2_3": 22,
    "4_1_1": 23,
    "1_33": 24,
    "5_15_5": 25,
    "3_27": 26,
    "1_15": 27,
    "4_1_2_1": 28,
    "6_3_1": 29,
    "8_1_1": 30,
    "6_7": 31,
    "5_15_3": 32,
    "7_3": 33,
    "1_19": 34,
    "6_4": 35,
    "8_1_4": 36,
    "8_8": 37,
    "1_16": 38,
    "1_11_1": 39,
    "6_6": 40,
    "5_15_1": 41,
    "7_2": 42,
    "5_15_2": 43,
    "7_12": 44,
    "3_18": 45,
    "5_6": 46,
    "5_5": 47,
    "7_4": 48,
    "4_1_2": 49,
    "8_2_2": 50,
    "7_11": 51,
    "1_22": 52,
    "1_27": 53,
    "2_3_2": 54,
    "5_15_2_2": 55,
    "1_8": 56,
    "3_13": 57,
    "2_3": 58,
    "8_3_3": 59,
    "2_3_3": 60,
    "7_7": 61,
    "1_11": 62,
    "8_13": 63,
    "1_12_2": 64,
    "1_20": 65,
    "1_12": 66,
    "3_32": 67,
    "2_5": 68,
    "3_1": 69,
    "4_8_2": 70,
    "3_20": 71,
    "3_2": 72,
    "2_3_6": 73,
    "5_22": 74,
    "5_18": 75,
    "2_3_5": 76,
    "7_5": 77,
    "8_4_1": 78,
    "3_14": 79,
    "1_2": 80,
    "1_20_2": 81,
    "4_1_4": 82,
    "7_6": 83,
    "8_1_3": 84,
    "8_3_1": 85,
    "4_3": 86,
    "4_1_5": 87,
    "8_2_3": 88,
    "8_2_4": 89,
    "1_31": 90,
    "3_10": 91,
    "4_2_2": 92,
    "7_1": 93,
    "3_28": 94,
    "4_1_3": 95,
    "5_4": 96,
    "5_3": 97,
    "6_8_2": 98,
    "3_31": 99,
    "6_2": 100,
    "1_21": 101,
    "3_21": 102,
    "1_13": 103,
    "1_14": 104,
    "2_3_4": 105,
    "4_8_3": 106,
    "6_15_2": 107,
    "2_6": 108,
    "3_18_2": 109,
    "4_1_2_2": 110,
    "1_7": 111,
    "3_19": 112,
    "1_18": 113,
    "2_7": 114,
    "8_5_4": 115,
    "5_15_7": 116,
    "5_14": 117,
    "5_21": 118,
    "1_1": 119,
    "6_15_1": 120,
    "8_6_4": 121,
    "8_15": 122,
    "4_5": 123,
    "3_11": 124,
    "8_18": 125,
    "8_4_4": 126,
    "3_30": 127,
    "5_7_1": 128,
    "5_7_2": 129,
    "1_5": 130,
    "3_29": 131,
    "6_15_3": 132,
    "5_12": 133,
    "3_16": 134,
    "1_30": 135,
    "5_11": 136,
    "1_6": 137,
    "8_6_2": 138,
    "6_8_3": 139,
    "3_12": 140,
    "3_33": 141,
    "8_4_3": 142,
    "5_8": 143,
    "8_14": 144,
    "8_17": 145,
    "3_6": 146,
    "1_26": 147,
    "8_5_2": 148,
    "6_8_1": 149,
    "5_17": 150,
    "1_10": 151,
    "8_16": 152,
    "7_18": 153,
    "7_14": 154,
    "8_23": 155,
}


def get_class_names_mapping() -> Dict[int, str]:
    """Create YOLO 0-based ID to GOST class name mapping."""
    id_to_gost = {v: k for k, v in LABEL_MAP.items()}
    class_names = {}
    for yolo_id in range(len(id_to_gost)):
        original_id = yolo_id + 1
        class_names[yolo_id] = id_to_gost.get(original_id, f"unknown_{original_id}")
    return class_names


def find_rtsd_structure() -> Dict[str, Path]:
    """Find RTSD dataset structure."""
    cwd = Path.cwd()
    print(f"Looking for RTSD dataset in {cwd}")
    print("=" * 80)

    rtsd_roots = []
    if (cwd / "data" / "raw" / "rtsd").exists():
        rtsd_roots.append(cwd / "data" / "raw" / "rtsd")
        print(f"Found: {rtsd_roots[-1].relative_to(cwd)}")

    for anno_file in cwd.rglob("anno.json"):
        root = anno_file.parent
        if root not in rtsd_roots and (root / "rtsd-frames").exists():
            rtsd_roots.append(root)
            print(f"Found: {root.relative_to(cwd)}")

    if not rtsd_roots:
        raise FileNotFoundError("RTSD dataset not found. Expected: data/raw/rtsd/")

    rtsd_root = rtsd_roots[0]
    print(f"\nUsing: {rtsd_root.relative_to(cwd)}")

    anno_files = {
        "train": rtsd_root / "train_anno.json",
        "val": rtsd_root / "val_anno.json",
        "anno": rtsd_root / "anno.json",
        "train_reduced": rtsd_root / "train_anno_reduced.json",
    }

    found_annos = {}
    for key, path in anno_files.items():
        if path.exists():
            found_annos[key] = path
            print(f"Found: {path.name}")

    images_dir = rtsd_root / "rtsd-frames"
    if not images_dir.exists():
        raise FileNotFoundError(f"rtsd-frames not found in {rtsd_root}")

    img_count = len(list(images_dir.glob("*")))
    print(f"rtsd-frames: {img_count} files")

    output_base_dir = cwd / "datasets" / "rtsd_yolo"
    return {
        "rtsd_root": rtsd_root,
        "annotations": found_annos,
        "images_dir": images_dir,
        "output_base_dir": output_base_dir,
    }


def select_annotation_file(anno_dict: Dict[str, Path]) -> Path:
    """Select annotation file by priority."""
    priority = ["train", "train_reduced", "anno", "val"]
    for key in priority:
        if key in anno_dict:
            print(f"\nUsing annotation: {anno_dict[key].name}")
            return anno_dict[key]
    raise ValueError("No suitable annotation files found")


def load_coco_data(json_file: Path) -> Tuple[Dict[int, str], List[Dict], List[Dict]]:
    """Load COCO format data."""
    logger.info(f"Loading COCO data from {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    image_id_to_name = {}
    for img in coco_data.get("images", []):
        image_id_to_name[img["id"]] = img["file_name"]

    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    logger.info(f"Loaded {len(image_id_to_name)} images")
    logger.info(f"Loaded {len(annotations)} annotations")
    logger.info(f"Loaded {len(categories)} categories")
    return image_id_to_name, annotations, categories


def bbox_to_yolo(
    x: float, y: float, width: float, height: float, img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """Convert COCO bbox to YOLO format."""
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
    """Get image dimensions."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Cannot read image {image_path}")
    return img.shape[0], img.shape[1]


def create_yolo_labels_and_load_dataset(
    annotations: List[Dict[str, Any]],
    image_id_to_name: Dict[int, str],
    images_dir: Path,
    labels_dir: Path,
) -> Tuple[Counter, List[Dict[str, Any]]]:
    """Create YOLO labels and load dataset."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_distribution = Counter()
    data_with_classes = []

    logger.info(f"Processing {len(annotations)} annotations")
    no_image_id = 0
    image_not_found = 0
    bbox_invalid = 0
    success_count = 0

    for idx, ann in tqdm(
        enumerate(annotations), desc="Processing annotations", total=len(annotations)
    ):
        try:
            image_id = ann.get("image_id")
            if image_id is None:
                no_image_id += 1
                continue

            image_name = image_id_to_name.get(image_id)
            if not image_name:
                image_not_found += 1
                continue

            class_id = ann.get("category_id")
            if class_id is None:
                continue

            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                bbox_invalid += 1
                continue

            x, y, bbox_width, bbox_height = bbox[:4]
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
                    matching = list(images_dir.glob(f"*/{Path(image_name).name}"))
                    if not matching:
                        matching = list(images_dir.glob(f"**/{Path(image_name).name}"))
                    if matching:
                        image_path = matching[0]

            if not image_path or not image_path.exists():
                image_not_found += 1
                continue

            img_height, img_width = get_image_dimensions(image_path)
            center_x, center_y, norm_width, norm_height = bbox_to_yolo(
                float(x),
                float(y),
                float(bbox_width),
                float(bbox_height),
                img_width,
                img_height,
            )

            label_file = labels_dir / f"{image_path.stem}.txt"
            with open(label_file, "w", encoding="utf-8") as f:
                f.write(
                    f"{class_id - 1} {center_x:.6f} {center_y:.6f} "
                    f"{norm_width:.6f} {norm_height:.6f}\n"
                )

            data_with_classes.append(
                {
                    "image": image_path,
                    "label": label_file,
                    "class_id": class_id - 1,
                }
            )

            class_distribution[class_id - 1] += 1
            success_count += 1

        except (IOError, ValueError, IndexError, TypeError) as exc:
            if idx < 5:
                logger.debug(f"Error processing annotation {idx}: {exc}")
            continue

    print("\n" + "=" * 80)
    print("Processing summary")
    print("=" * 80)
    print(f"Total annotations:       {len(annotations)}")
    print(f"Successfully processed:  {success_count}")
    print(f"No image_id:             {no_image_id}")
    print(f"Image not found:         {image_not_found}")
    print(f"Invalid bbox:            {bbox_invalid}")
    print(f"Total skipped:           {len(annotations) - success_count}")
    print("=" * 80)

    if success_count == 0:
        raise ValueError("No valid samples found. Check image paths.")

    return class_distribution, data_with_classes


def print_class_statistics(
    class_distribution: Counter, display_count: int = 20
) -> None:
    """Print top classes statistics."""
    print("\n" + "=" * 80)
    print(f"Top {min(display_count, len(class_distribution))} classes")
    print("=" * 80)

    sorted_classes = class_distribution.most_common(display_count)
    for rank, (class_id, count) in enumerate(sorted_classes, 1):
        bar = "█" * (count // 100) if count > 0 else ""
        gost_name = get_class_names_mapping().get(class_id, "unknown")
        print(f"  {rank:2d}. {gost_name:13s}: {count:6d} images {bar}")


def filter_rare_classes(
    data_with_classes: List[Dict[str, Any]],
    class_distribution: Counter,
    min_samples: int = 5,
) -> Tuple[List[Dict[str, Any]], Counter, int]:
    """Filter rare classes before stratified split."""
    print("\n" + "=" * 80)
    print(f"Filtering rare classes (min_samples={min_samples})")
    print("=" * 80)

    rare_classes = {
        cid for cid, count in class_distribution.items() if count < min_samples
    }

    if not rare_classes:
        print("All classes have sufficient samples - filtering not needed")
        return data_with_classes, class_distribution, 0

    print(f"Found {len(rare_classes)} rare classes with < {min_samples} samples")
    affected = sum(class_distribution[cid] for cid in rare_classes)
    print(f"Affected samples: {affected}")

    filtered_data = [
        item for item in data_with_classes if item["class_id"] not in rare_classes
    ]

    filtered_distribution = Counter()
    for item in filtered_data:
        filtered_distribution[item["class_id"]] += 1

    skipped = len(data_with_classes) - len(filtered_data)
    pct = skipped / len(data_with_classes) * 100

    print(f"\nAfter filtering:")
    print(f"  Samples:  {len(data_with_classes)} -> {len(filtered_data)}")
    print(f"           (removed {skipped}, {pct:.1f}%)")
    print(f"  Classes:  {len(class_distribution)} -> {len(filtered_distribution)}")

    return filtered_data, filtered_distribution, skipped


def copy_split_files(
    data_list: List[Dict[str, Any]], split_name: str, output_base_dir: Path
) -> None:
    """Copy files to split directories."""
    img_dir = output_base_dir / "images" / split_name
    lbl_dir = output_base_dir / "labels" / split_name

    for item in tqdm(data_list, desc=f"Copying {split_name}"):
        shutil.copy(str(item["image"]), str(img_dir / item["image"].name))
        shutil.copy(str(item["label"]), str(lbl_dir / item["label"].name))


def verify_split_integrity(output_base_dir: Path) -> bool:
    """Verify image-label pairs integrity."""
    print("=" * 80)

    all_valid = True
    for split in ["train", "val", "test"]:
        img_dir = output_base_dir / "images" / split
        lbl_dir = output_base_dir / "labels" / split

        img_count = len(list(img_dir.glob("*")))
        lbl_count = len(list(lbl_dir.glob("*.txt")))

        status = "OK" if img_count == lbl_count else "MISMATCH"
        print(f"  {split:5s}: {img_count:6d} images, {lbl_count:6d} labels [{status}]")

        if img_count != lbl_count:
            all_valid = False

    print("=" * 80)
    return all_valid


def save_data_yaml(class_names: Dict[int, str], output_base_dir: Path) -> None:
    """Save data.yaml with correct class names."""
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

    print(f"Saved: {yaml_path}")


def save_split_statistics(
    data_with_classes: List[Dict[str, Any]],
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    class_distribution: Counter,
    skipped: int,
    output_base_dir: Path,
) -> None:
    """Save split statistics to JSON."""
    total_samples = len(data_with_classes)

    split_stats = {
        "total_samples": total_samples,
        "skipped_rare_classes": skipped,
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
        "dataset_type": "RTSD - Russian Traffic Sign Dataset (COCO Format)",
    }

    stats_file = output_base_dir / "split_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)

    print(f"Statistics saved: {stats_file}")


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
    """Main function to split dataset into train/val/test."""
    print("\n" + "=" * 80)
    print("RTSD Dataset Stratified Split")
    print("=" * 80)

    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    np.random.seed(random_seed)

    for split in ["train", "val", "test"]:
        (output_base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("\nLoading COCO data...")
    image_id_to_name, annotations, categories = load_coco_data(anno_file)

    print("\nCreating YOLO labels...")
    temp_labels_dir = output_base_dir / "labels" / "temp"
    class_distribution, data_with_classes = create_yolo_labels_and_load_dataset(
        annotations, image_id_to_name, images_dir, temp_labels_dir
    )

    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(data_with_classes)}")
    print(f"  Unique classes: {len(class_distribution)}")
    print_class_statistics(class_distribution)

    print("\nFiltering rare classes before split...")
    data_with_classes, class_distribution, skipped = filter_rare_classes(
        data_with_classes, class_distribution, min_samples=min_samples_per_class
    )

    print("\n" + "=" * 80)
    print("Performing stratified split")
    print("=" * 80)
    print(f"  Train: {train_ratio*100:.1f}%")
    print(f"  Val:   {val_ratio*100:.1f}%")
    print(f"  Test:  {test_ratio*100:.1f}%")

    stratify_labels = [item["class_id"] for item in data_with_classes]

    train_val_data, test_data = train_test_split(
        data_with_classes,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify_labels,
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_val_labels = [item["class_id"] for item in train_val_data]
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=train_val_labels,
    )

    total_samples = len(data_with_classes)

    print("\n" + "=" * 80)
    print("Split summary")
    print("=" * 80)
    pct_train = len(train_data) / total_samples * 100
    pct_val = len(val_data) / total_samples * 100
    pct_test = len(test_data) / total_samples * 100
    print(f"  Train: {len(train_data):6d} ({pct_train:5.1f}%)")
    print(f"  Val:   {len(val_data):6d} ({pct_val:5.1f}%)")
    print(f"  Test:  {len(test_data):6d} ({pct_test:5.1f}%)")
    print(f"  Total: {total_samples:6d}")

    print("\nCopying files to output directories...")
    copy_split_files(train_data, "train", output_base_dir)
    copy_split_files(val_data, "val", output_base_dir)
    copy_split_files(test_data, "test", output_base_dir)

    print("\nVerifying split integrity...")
    verify_split_integrity(output_base_dir)

    print("\nSaving statistics...")
    save_split_statistics(
        data_with_classes,
        train_data,
        val_data,
        test_data,
        class_distribution,
        skipped,
        output_base_dir,
    )

    print("\nSaving data.yaml with correct class names...")
    class_names = get_class_names_mapping()
    save_data_yaml(class_names, output_base_dir)

    if temp_labels_dir.exists():
        shutil.rmtree(temp_labels_dir)

    return {
        "total_samples": total_samples,
        "train": {
            "count": len(train_data),
            "percentage": pct_train,
        },
        "val": {
            "count": len(val_data),
            "percentage": pct_val,
        },
        "test": {
            "count": len(test_data),
            "percentage": pct_test,
        },
        "classes": len(class_distribution),
        "skipped_rare_classes": skipped,
    }


def main() -> None:
    """Entry point."""
    try:
        print("\n" + "=" * 80)
        print("RTSD to YOLO Converter")
        print("=" * 80)

        paths = find_rtsd_structure()
        anno_file = select_annotation_file(paths["annotations"])

        result = stratified_split(
            anno_file=anno_file,
            images_dir=paths["images_dir"],
            output_base_dir=paths["output_base_dir"],
            min_samples_per_class=MIN_SAMPLES_PER_CLASS,
        )

        print("\n" + "=" * 80)
        print("Conversion completed successfully")
        print("=" * 80)
        print(f"Output: {paths['output_base_dir'].relative_to(Path.cwd())}/")
        print(f"Total samples: {result['total_samples']}")
        print(f"Classes: {result['classes']}")
        print("\n" + "=" * 80)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as error:
        logger.error(f"Failed: {error}", exc_info=True)
        print(f"\nERROR: {error}")
        raise


if __name__ == "__main__":
    main()
