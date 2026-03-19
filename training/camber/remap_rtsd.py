#!/usr/bin/env python3
"""
RTSD COCO JSON → YOLO format remapper для белорусских знаков.

Что делает:
  1. Читает оригинальный RTSD датасет в формате COCO JSON
  2. Перемаппирует class_id RTSD → новый class_id белорусских знаков
  3. Исключает знаки без белорусского эквивалента (RUSSIAN_ONLY)
  4. Конвертирует bbox COCO (x,y,w,h абс.) → YOLO (cx,cy,w,h норм.)
  5. Сохраняет .txt аннотации + data.yaml + class_mapping.py

Использование:
  python remap_rtsd.py \\
      --coco_dir /path/to/rtsd_coco \\
      --output_dir /path/to/rtsd_yolo_belarus \\
      --mapping_json belarus_mapping_data.json

Ожидаемая структура --coco_dir:
  rtsd_coco/
  ├── train/
  │   ├── images/
  │   └── annotations.json
  ├── val/
  │   ├── images/
  │   └── annotations.json
  └── test/           (опционально)
      ├── images/
      └── annotations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("remap_rtsd")


# ── Helpers ────────────────────────────────────────────────────────────────────

def coco_bbox_to_yolo(
    bbox: list[float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert COCO [x, y, w, h] (absolute pixels) → YOLO [cx, cy, w, h] (0-1)."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def load_mapping(mapping_json: Path) -> dict:
    with open(mapping_json, encoding="utf-8") as f:
        return json.load(f)


# ── Core converter ─────────────────────────────────────────────────────────────

def convert_split(
    split_dir: Path,
    output_dir: Path,
    rtsd_id_to_by: dict[str, str],   # "118" → "1.1"
    by_to_new_id: dict[str, int],     # "1.1" → 0
    split_name: str,
) -> dict:
    """Convert one split (train/val/test)."""
    ann_file = split_dir / "annotations.json"
    img_dir = split_dir / "images"

    if not ann_file.exists():
        logger.warning("Нет %s — пропускаем сплит '%s'", ann_file, split_name)
        return {}

    logger.info("Обрабатываем сплит: %s", split_name)

    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup: image_id → image info
    images: dict[int, dict] = {img["id"]: img for img in coco["images"]}

    # Build lookup: image_id → list of annotations
    ann_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    # RTSD COCO category_id → RTSD numeric ID
    # In RTSD dataset category_id starts from 0 or 1 depending on version
    # We check if category "name" matches RTSD_ID from our mapping
    # Categories in RTSD COCO: {"id": X, "name": "X"} where name == rtsd_label_index
    cat_id_to_rtsd_id: dict[int, int] = {}
    for cat in coco.get("categories", []):
        try:
            rtsd_id = int(cat["name"])
            cat_id_to_rtsd_id[cat["id"]] = rtsd_id
        except (ValueError, KeyError):
            # Some versions use category name as sign code like "2_1" or "1.1"
            cat_id_to_rtsd_id[cat["id"]] = cat["id"]

    # Output dirs
    out_img_dir = output_dir / split_name / "images"
    out_lbl_dir = output_dir / split_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": len(images),
        "written_images": 0,
        "skipped_images": 0,
        "total_annotations": len(coco["annotations"]),
        "written_annotations": 0,
        "skipped_annotations_no_equiv": 0,
    }

    for img_id, img_info in images.items():
        img_filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        annotations = ann_by_image.get(img_id, [])
        yolo_lines = []

        for ann in annotations:
            cat_id = ann["category_id"]
            rtsd_numeric_id = cat_id_to_rtsd_id.get(cat_id, cat_id)

            # Look up Belarus equivalent
            by_code = rtsd_id_to_by.get(str(rtsd_numeric_id))
            if by_code is None:
                stats["skipped_annotations_no_equiv"] += 1
                continue

            new_class_id = by_to_new_id.get(by_code)
            if new_class_id is None:
                stats["skipped_annotations_no_equiv"] += 1
                continue

            bbox = ann["bbox"]
            if img_w == 0 or img_h == 0:
                continue

            cx, cy, nw, nh = coco_bbox_to_yolo(bbox, img_w, img_h)

            # Skip degenerate boxes
            if nw < 0.001 or nh < 0.001:
                continue

            yolo_lines.append(f"{new_class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats["written_annotations"] += 1

        # Copy image
        src_img = split_dir / "images" / Path(img_filename).name
        if not src_img.exists():
            # Try direct path
            src_img = img_dir / img_filename
        if not src_img.exists():
            stats["skipped_images"] += 1
            continue

        dst_img = out_img_dir / src_img.name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Write label file (even if empty — YOLO needs it for background images)
        lbl_file = out_lbl_dir / (src_img.stem + ".txt")
        lbl_file.write_text("\n".join(yolo_lines), encoding="utf-8")

        stats["written_images"] += 1

    logger.info(
        "  %s: %d/%d изображений, %d аннотаций (%d пропущено без эквивалента)",
        split_name,
        stats["written_images"],
        stats["total_images"],
        stats["written_annotations"],
        stats["skipped_annotations_no_equiv"],
    )
    return stats


# ── data.yaml writer ───────────────────────────────────────────────────────────

def write_data_yaml(
    output_dir: Path,
    classes: list[str],
    names_ru: dict[str, str],
    splits_present: list[str],
) -> None:
    """Write data.yaml for YOLO training."""
    lines = ["# YOLO dataset config — RTSD remapped to Belarus road signs",
             "# Generated by remap_rtsd.py", ""]

    for split in ["train", "val", "test"]:
        if split in splits_present:
            lines.append(f"{split}: {split}/images")

    lines += [
        "",
        f"nc: {len(classes)}  # number of classes",
        "",
        "names:  # BY code: Russian name",
    ]
    for code in classes:
        name = names_ru.get(code, code)
        lines.append(f"  - '{code}: {name}'")

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Записан data.yaml: %s  (%d классов)", yaml_path, len(classes))


# ── class_mapping.py writer ────────────────────────────────────────────────────

def write_class_mapping_py(
    output_dir: Path,
    rtsd_id_to_by: dict[str, str],
    by_to_new_id: dict[str, int],
    classes: list[str],
    names_ru: dict[str, str],
) -> None:
    """Write updated class_mapping.py for RaspberryRoadSign inference."""
    # Build RTSD_ID → new_class_id (what the model actually outputs)
    rtsd_to_new = {}
    for rtsd_id_str, by_code in rtsd_id_to_by.items():
        new_id = by_to_new_id.get(by_code)
        if new_id is not None:
            rtsd_to_new[int(rtsd_id_str)] = new_id

    # Build new_class_id → BY code (for display)
    new_id_to_by = {v: k for k, v in by_to_new_id.items()}

    lines = [
        '"""',
        "RTSD to Belarusian traffic sign class mapping.",
        "Auto-generated by remap_rtsd.py from rtsd_belarus_mapping_verified.csv",
        '"""',
        "",
        "from typing import Dict, Optional",
        "",
        "# New class ID (0-based, as output by fine-tuned model) → BY sign code",
        "NEW_ID_TO_BELARUSIAN: Dict[int, str] = {",
    ]
    for new_id in sorted(new_id_to_by.keys()):
        by_code = new_id_to_by[new_id]
        name = names_ru.get(by_code, "")
        lines.append(f"    {new_id}: \"{by_code}\",  # {name}")
    lines += ["}", "", "# BY sign code → Russian name", "BY_NAMES_RU: Dict[str, str] = {"]
    for code in classes:
        name = names_ru.get(code, "")
        lines.append(f"    \"{code}\": \"{name}\",")
    lines += [
        "}",
        "",
        "# Number of classes in fine-tuned model",
        f"NUM_CLASSES: int = {len(classes)}",
        "",
        "",
        "class ClassMapper:",
        '    """Utility class for mapping between class IDs and sign codes."""',
        "",
        "    @staticmethod",
        "    def id_to_belarusian(class_id: int) -> Optional[str]:",
        '        """Convert new model class ID → Belarusian sign code."""',
        "        return NEW_ID_TO_BELARUSIAN.get(class_id)",
        "",
        "    @staticmethod",
        "    def get_name_ru(by_code: str) -> str:",
        '        """Get Russian name for a BY sign code."""',
        "        return BY_NAMES_RU.get(by_code, by_code)",
        "",
        "    @staticmethod",
        "    def get_num_classes() -> int:",
        '        """Total number of classes."""',
        "        return NUM_CLASSES",
        "",
        "    @staticmethod",
        "    def get_all_mappings() -> Dict[int, str]:",
        "        return NEW_ID_TO_BELARUSIAN.copy()",
    ]

    out = output_dir / "class_mapping.py"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Записан class_mapping.py: %s", out)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap RTSD COCO dataset to Belarus YOLO format"
    )
    parser.add_argument(
        "--coco_dir",
        type=Path,
        required=True,
        help="Path to RTSD COCO root (contains train/, val/, test/ subfolders)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("rtsd_yolo_belarus"),
        help="Output directory for remapped YOLO dataset",
    )
    parser.add_argument(
        "--mapping_json",
        type=Path,
        default=Path("belarus_mapping_data.json"),
        help="Path to belarus_mapping_data.json (generated by remap_rtsd.py setup)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process",
    )
    args = parser.parse_args()

    # Load mapping
    mapping = load_mapping(args.mapping_json)
    rtsd_id_to_by: dict[str, str] = mapping["rtsd_id_to_by"]
    by_to_new_id: dict[str, int] = mapping["by_to_new_id"]
    classes: list[str] = mapping["classes"]
    names_ru: dict[str, str] = mapping["names_ru"]

    logger.info(
        "Маппинг загружен: %d RTSD ID → %d уникальных BY классов",
        len(rtsd_id_to_by),
        len(classes),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits_done = []
    for split in args.splits:
        split_dir = args.coco_dir / split
        if not split_dir.exists():
            logger.warning("Сплит не найден: %s", split_dir)
            continue
        convert_split(
            split_dir=split_dir,
            output_dir=args.output_dir,
            rtsd_id_to_by=rtsd_id_to_by,
            by_to_new_id=by_to_new_id,
            split_name=split,
        )
        splits_done.append(split)

    # Write data.yaml
    write_data_yaml(args.output_dir, classes, names_ru, splits_done)

    # Write updated class_mapping.py
    write_class_mapping_py(
        args.output_dir, rtsd_id_to_by, by_to_new_id, classes, names_ru
    )

    logger.info("✅ Готово! Датасет: %s", args.output_dir)
    logger.info(
        "   Классов: %d | data.yaml: %s/data.yaml",
        len(classes),
        args.output_dir,
    )
    logger.info("   Следующий шаг: загрузить на Camber Stash — см. upload_to_camber.py")


if __name__ == "__main__":
    main()
