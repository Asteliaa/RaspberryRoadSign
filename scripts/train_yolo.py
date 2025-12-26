import argparse
import json
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def setup_gpu() -> int:
    """Setup GPU for training. Returns device ID or -1 for CPU."""
    print("GPU Setup:")

    if not torch.cuda.is_available():
        print("  CUDA not available → Using CPU")
        return -1

    gpu_count = torch.cuda.device_count()
    device_id = 0
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_memory_gb = torch.cuda.get_device_properties(
        device_id).total_memory / (1024 ** 3)

    print(f"  Device: {gpu_name} ({gpu_memory_gb:.2f} GB)")

    # Test GPU
    try:
        test_tensor = torch.randn(100, 100).to(f"cuda:{device_id}")
        result = torch.matmul(test_tensor, test_tensor)
        del test_tensor, result
        torch.cuda.empty_cache()
        print(f"  ✓ GPU ready\n")
    except RuntimeError:
        print("  ✗ GPU test failed → Using CPU\n")
        return -1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    return device_id


def find_config_file(config_path: str | Path) -> Path:
    """Find config file with intelligent path resolution"""
    config_path = Path(config_path)

    if config_path.is_absolute():
        if config_path.exists():
            return config_path
        raise FileNotFoundError(f"Config not found: {config_path}")

    config_relative = PROJECT_ROOT / config_path
    if config_relative.exists():
        return config_relative

    for location in [
        PROJECT_ROOT / "configs" / config_path.name,
        PROJECT_ROOT / "scripts" / config_path.name,
        PROJECT_ROOT / config_path.name,
    ]:
        if location.exists():
            return location

    raise FileNotFoundError(f"Config not found: {config_path}")


def load_training_config(config_path: Path | str) -> Dict[str, Any]:
    """Load training configuration from YAML file"""
    config_path = find_config_file(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate required configuration parameters"""
    required_keys = ["model", "data", "epochs", "batch"]
    return all(key in config for key in required_keys)


def check_dataset(config: Dict[str, Any]) -> bool:
    """Verify dataset structure and availability"""
    data_yaml = config.get("data")
    data_yaml_path = Path(data_yaml)

    if not data_yaml_path.is_absolute():
        data_yaml_path = PROJECT_ROOT / data_yaml

    if not data_yaml_path.exists():
        print(f"✗ Dataset config not found: {data_yaml_path}")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    dataset_path = Path(data_config.get("path", ""))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    print("Dataset Check:")
    all_valid = True
    for split in ["train", "val"]:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  ✗ {split}: MISSING")
            all_valid = False
            continue

        img_count = len(list(img_dir.glob("*")))
        lbl_count = len(list(lbl_dir.glob("*.txt")))
        status = "✓" if img_count == lbl_count else "✗"
        print(f"  {status} {split}: {img_count} images, {lbl_count} labels")

    print()
    return all_valid


def train(
    config_path: Path | str,
    device_id: int,
    resume: bool = False
) -> Optional[Any]:
    """Execute model training"""
    try:
        config = load_training_config(config_path)
    except FileNotFoundError as e:
        print(f"✗ Config error: {e}\n")
        return None

    if not validate_config(config):
        print("✗ Config validation failed\n")
        return None

    print("Config:")
    for key, value in config.items():
        if isinstance(value, (dict, list)) and len(str(value)) > 80:
            print(f"  {key}: <dict/list>")
        else:
            print(f"  {key}: {value}")
    print()

    if not check_dataset(config):
        print("⚠ Some dataset splits are missing\n")

    print("Model Initialization:")
    model_name = config.get("model")
    print(f"  Loading: {model_name}")

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"✗ Model loading failed: {e}\n")
        return None

    try:
        total_params = sum(p.numel() for p in model.model.parameters()) # type: ignore
        trainable_params = sum(p.numel()
                               for p in model.model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}\n")
    except Exception:
        pass

    print("=" * 60)
    print("Training")
    print("=" * 60)

    training_start = time.time()
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config.get('name', 'unnamed')}")
    print(f"Epochs: {config.get('epochs')}, Batch: {config.get('batch')}")
    device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"
    print(f"Device: {device_str}\n")

    if device_id >= 0:
        config['device'] = device_id

    try:
        results = model.train(**config)

        training_time = (time.time() - training_start) / 3600

        print("\n" + "=" * 60)
        print("✓ Training Complete")
        print("=" * 60)
        print(f"Duration: {training_time:.2f}h")
        print(f"Results: {results.save_dir}") # type: ignore
        print(f"Model: {results.save_dir}/weights/best.pt\n") # type: ignore

        _save_metrics(config, results, training_time)
        _create_visualizations(results)

        return results

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user\n")
        return None

    except Exception as e:
        print(f"\n✗ Training failed: {e}\n")
        traceback.print_exc()
        return None


def _save_metrics(config: Dict[str, Any], results: Any, training_time: float) -> None:
    """Save training metrics to JSON"""
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


def _create_visualizations(results: Any) -> None:
    """Create training visualization plots"""
    results_csv = Path(results.save_dir) / "results.csv"

    if not results_csv.exists():
        return

    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
            axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="Train")
            axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="Val")
            axes[0, 0].set_title("Box Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
            axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="Train")
            axes[0, 1].plot(df["epoch"], df["val/cls_loss"], label="Val")
            axes[0, 1].set_title("Classification Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        if "metrics/mAP50(B)" in df.columns:
            axes[1, 0].plot(df["epoch"], df["metrics/mAP50(B)"], color="green")
            axes[1, 0].set_title("mAP@0.5")
            axes[1, 0].grid(True, alpha=0.3)

        if "metrics/precision(B)" in df.columns:
            axes[1, 1].plot(
                df["epoch"], df["metrics/precision(B)"], label="Precision")
            axes[1, 1].plot(
                df["epoch"], df["metrics/recall(B)"], label="Recall")
            axes[1, 1].set_title("Precision & Recall")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Training Results", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_file = Path(results.save_dir) / "training_curves.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

    except Exception:
        pass


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Training Pipeline",
        epilog="Example: python train_yolo_gpu.py --config configs/training_nano.yaml"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_nano.yaml",
        help="Path to training config YAML",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point"""
    device_id = setup_gpu()
    args = parse_arguments()

    try:
        results = train(args.config, device_id=device_id, resume=args.resume)

        if results:
            print("✓ Training Successful")
            print(f"  Results: {results.save_dir}")
            print(
                f"  Export: python scripts/export_models.py --model {results.save_dir}/weights/best.pt\n")
        else:
            print("✗ Training Failed\n")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Fatal error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
