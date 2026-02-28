#!/usr/bin/env python3
"""CLI script for traffic sign detection in videos."""

import sys
import argparse
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from RaspberryRoadSign.inference.detector import TrafficSignDetector
from RaspberryRoadSign.utils.logging import setup_logging


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Detect traffic signs in video files"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Input video file path"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output video file path"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (0-1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, name="infer")
    logger = logging.getLogger("infer")
    
    try:
        # Initialize detector
        detector = TrafficSignDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            device=args.device
        )
        
        # Run detection
        stats = detector.detect_video(args.video, args.output)
        
        # Print results
        logger.info("Detection completed!")
        logger.info(f"Total detections: {stats['detections']}")
        logger.info(f"Processed frames: {stats['total_frames']}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
