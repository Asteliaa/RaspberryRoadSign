#!/usr/bin/env python3
"""CLI script for YOLO model training."""

import sys
import argparse
import yaml
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from RaspberryRoadSign.training.trainer import TrainingPipeline
from RaspberryRoadSign.utils.logging import setup_logging


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model on traffic sign dataset"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to training config (YAML)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, name="train")
    logger = logging.getLogger("train")
    
    try:
        # Load config
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
        
        # Extract parameters
        model = config.get('model', 'yolov11n')
        data_path = Path(config.get('data', 'datasets/rtsd_yolo/data.yaml'))
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 24)
        imgsz = config.get('imgsz', 480)
        device = config.get('device', '0')
        
        # Initialize trainer
        trainer = TrainingPipeline(model=model, device=device)
        
        # Train
        logger.info("Starting training...")
        results = trainer.train(
            data_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {results.get('model_path')}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
