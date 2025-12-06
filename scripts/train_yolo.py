"""
YOLOv11 training pipeline - Simplified & Config-Driven.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def find_config_file(config_path: str | Path) -> Path:
    """Найти конфиг-файл с умным поиском."""
    config_path = Path(config_path)

    # Абсолютный путь
    if config_path.is_absolute():
        if config_path.exists():
            return config_path
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Относительно PROJECT_ROOT
    config_relative = PROJECT_ROOT / config_path
    if config_relative.exists():
        return config_relative

    # Стандартные места
    for location in [
        PROJECT_ROOT / "configs" / config_path.name,
        PROJECT_ROOT / "scripts" / config_path.name,
        PROJECT_ROOT / config_path.name,
    ]:
        if location.exists():
            return location

    raise FileNotFoundError(f"Config not found: {config_path}")


def load_training_config(config_path: Path | str) -> Dict[str, Any]:
    """Загрузить конфиг из YAML."""
    config_path = find_config_file(config_path)
    logger.info(f"Loading config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Проверить обязательные параметры в конфиге."""
    required_keys = ["model", "data", "epochs", "batch"]

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False

    return True


def check_environment() -> bool:
    """Проверить PyTorch, CUDA, GPU."""
    print("\n" + "=" * 80)
    print("Environment Check")
    print("=" * 80)

    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            total_mem = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            print(f"  GPU {idx}: {name} ({total_mem:.2f} GB)")
        torch.cuda.empty_cache()
    else:
        logger.warning("CUDA not available - training will be slow on CPU")

    return True


def check_dataset(config: Dict[str, Any]) -> bool:
    """Проверить существование датасета."""
    data_yaml = config.get("data")
    data_yaml_path = Path(data_yaml)

    if not data_yaml_path.is_absolute():
        data_yaml_path = PROJECT_ROOT / data_yaml

    if not data_yaml_path.exists():
        logger.error(f"Data config not found: {data_yaml_path}")
        print(f"❌ Dataset not found: {data_yaml_path}")
        return False

    # Проверить структуру датасета
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    dataset_path = Path(data_config.get("path", ""))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    print(f"\nDataset Check")
    print("-" * 80)
    print(f"Data config: {data_yaml_path}")
    print(f"Dataset path: {dataset_path}")

    all_valid = True
    for split in ["train", "val"]:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            logger.warning(f"Split missing: {split}")
            print(f"  {split}: ❌ MISSING")
            all_valid = False
            continue

        img_count = len(list(img_dir.glob("*")))
        lbl_count = len(list(lbl_dir.glob("*.txt")))
        status = "✅" if img_count == lbl_count else "⚠️"
        print(f"  {split}: {img_count} images, {lbl_count} labels {status}")

    return all_valid


def train(config_path: Path | str, resume: bool = False) -> Optional[Any]:
    """Главная функция обучения."""
    print("=" * 80)
    print("YOLOv11 Training - Config-Driven")
    print("=" * 80)
    print(f"Working directory: {PROJECT_ROOT}\n")

    # Загрузить конфиг
    try:
        config = load_training_config(config_path)
    except FileNotFoundError as e:
        logger.error(f"Config error: {e}")
        print(f"❌ {e}")
        return None

    # Валидировать конфиг
    if not validate_config(config):
        logger.error("Config validation failed")
        return None

    # Показать конфиг
    print("\nConfig loaded:")
    print("-" * 80)
    for key, value in config.items():
        # Скрыть большие списки
        if isinstance(value, dict) and len(str(value)) > 100:
            print(f"  {key}: <dict with {len(value)} items>")
        elif isinstance(value, list) and len(str(value)) > 100:
            print(f"  {key}: <list with {len(value)} items>")
        else:
            print(f"  {key}: {value}")

    # Проверить окружение
    if not check_environment():
        logger.error("Environment check failed")
        return None

    # Проверить датасет
    if not check_dataset(config):
        logger.warning("Some dataset splits are missing")

    # Инициализировать модель
    print("\n" + "=" * 80)
    print("Model Initialization")
    print("=" * 80)

    model_name = config.get("model")
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # Показать информацию о модели
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    except Exception as e:
        logger.debug(f"Could not display model info: {e}")

    # Обучение
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    training_start = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config.get('name', 'unnamed')}")
    print(f"Epochs: {config.get('epochs')}")
    print(f"Batch: {config.get('batch')}")
    print(f"Fraction: {config.get('fraction', 1.0)}")

    try:
        # ГЛАВНОЕ: Передать весь конфиг в YOLO
        # Конфиг уже содержит все параметры!
        results = model.train(**config)

        training_time = (time.time() - training_start) / 3600

        print("\n" + "=" * 80)
        print("Training Completed")
        print("=" * 80)
        print(f"Duration: {training_time:.2f} hours")
        print(f"Results: {results.save_dir}")

        # Постобработка
        _save_metrics(config, results, training_time)
        _create_visualizations(results)

        return results

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\n❌ Training interrupted")
        return None

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return None


def _save_metrics(config: Dict[str, Any], results: Any, training_time: float) -> None:
    """Сохранить метрики в JSON."""
    metrics_file = Path(results.save_dir) / "training_metrics.json"

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "training_time_hours": training_time,
        "config": config,
        "save_dir": str(results.save_dir),
        "best_model": str(Path(results.save_dir) / "weights" / "best.pt"),
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics saved: {metrics_file}")


def _create_visualizations(results: Any) -> None:
    """Создать графики обучения."""
    results_csv = Path(results.save_dir) / "results.csv"

    if not results_csv.exists():
        logger.warning("Results CSV not found")
        return

    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Box loss
        if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
            axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="Train")
            axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="Val")
            axes[0, 0].set_title("Box Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Classification loss
        if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
            axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="Train")
            axes[0, 1].plot(df["epoch"], df["val/cls_loss"], label="Val")
            axes[0, 1].set_title("Classification Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # mAP
        if "metrics/mAP50(B)" in df.columns:
            axes[1, 0].plot(df["epoch"], df["metrics/mAP50(B)"], color="green")
            axes[1, 0].set_title("mAP@0.5")
            axes[1, 0].grid(True, alpha=0.3)

        # Precision & Recall
        if "metrics/precision(B)" in df.columns:
            axes[1, 1].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
            axes[1, 1].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
            axes[1, 1].set_title("Precision & Recall")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Training Results", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_file = Path(results.save_dir) / "training_curves.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_file}")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def parse_arguments() -> argparse.Namespace:
    """Парсить аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Training - Config-Driven",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default config
  python 07_train_yolo.py

  # Custom config
  python 07_train_yolo.py --config configs/training_config_rtsd.yaml

  # Resume training5
  python 07_train_yolo.py --config configs/training_config_rtsd.yaml --resume

  # Absolute path
  python train_yolo.py --config /path/to/config.yaml

All parameters are configured in YAML - edit config file to change anything!
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config_rtsd.yaml",
        help="Path to training config YAML",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_arguments()

    try:
        results = train(args.config, resume=args.resume)

        if results:
            print("\n" + "=" * 80)
            print("✅ Training Successful")
            print("=" * 80)
            print(f"Results: {results.save_dir}")
            print(f"Best model: {results.save_dir}/weights/best.pt")
        else:
            print("\n" + "=" * 80)
            print("❌ Training Failed")
            print("=" * 80)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
