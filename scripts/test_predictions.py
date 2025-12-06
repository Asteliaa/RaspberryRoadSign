"""
Test trained model on image directory.

Runs inference on a directory of images and generates annotated
predictions with confidence scores and detection statistics.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ImageTester:
    """
    Test YOLOv11 model on image dataset.

    Performs inference on directory of images, annotates results,
    and computes performance statistics.
    """

    def __init__(
        self,
        model_path: Path | str,
        images_dir: Path | str,
        output_dir: Path | str = "results/predictions",
    ):
        """
        Initialize image tester.

        Args:
            model_path: Path to trained model (.pt file).
            images_dir: Directory containing test images.
            output_dir: Directory for annotated results.

        Raises:
            FileNotFoundError: If model or images directory not found.
        """
        self.model_path = Path(model_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not self.images_dir.exists():
            msg = f"Images directory not found: {self.images_dir}"
            raise FileNotFoundError(msg)

        self.model = None
        self.results_stats = {
            "total_images": 0,
            "total_time": 0.0,
            "total_detections": 0,
            "images_with_detections": 0,
        }

    def load_model(self) -> YOLO:
        """
        Load YOLO model from checkpoint.

        Returns:
            Loaded YOLO model.
        """
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        return self.model

    def collect_images(self) -> List[Path]:
        """
        Collect all image files from directory.

        Returns:
            List of image file paths.
        """
        supported_formats = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

        image_files = []
        for fmt in supported_formats:
            image_files.extend(self.images_dir.glob(fmt))

        return sorted(image_files)

    def run_inference(
        self, image_path: Path, conf: float, imgsz: int
    ) -> Tuple[any, float]:
        """
        Run inference on single image.

        Args:
            image_path: Path to image file.
            conf: Confidence threshold.
            imgsz: Image size for inference.

        Returns:
            Tuple of (results, inference_time_seconds).
        """
        start_time = time.time()
        results = self.model(str(image_path), conf=conf, imgsz=imgsz)
        inference_time = time.time() - start_time

        return results, inference_time

    def process_detections(self, results: any) -> List[Dict]:
        """
        Extract detection information from results.

        Args:
            results: YOLO inference results.

        Returns:
            List of detection dictionaries.
        """
        detections = []

        result = results[0]

        for box in result.boxes:
            detection = {
                "class_id": int(box.cls[0]),
                "class_name": self.model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            }
            detections.append(detection)

        return detections

    def save_annotated_image(self, results: any, image_path: Path) -> bool:
        """
        Save annotated image with predictions.

        Args:
            results: YOLO inference results.
            image_path: Original image path.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            result = results[0]
            annotated_image = result.plot()

            output_path = self.output_dir / image_path.name
            success = cv2.imwrite(str(output_path), annotated_image)

            return success

        except Exception as error:
            logger.error(f"Failed to save annotated image: {error}")
            return False

    def test_on_directory(self, conf: float = 0.25, imgsz: int = 320) -> Dict:
        """
        Test model on all images in directory.

        Args:
            conf: Confidence threshold for detections.
            imgsz: Image size for inference.

        Returns:
            Dictionary with test statistics.
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Collect images
        image_files = self.collect_images()

        if not image_files:
            logger.error("No images found in directory")
            return self.results_stats

        print("\n" + "=" * 80)
        print("Testing Model on Images")
        print("=" * 80)
        print(f"Found images: {len(image_files)}")
        print(f"Confidence threshold: {conf}")
        print(f"Image size: {imgsz}\n")

        # Process each image
        for idx, img_file in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing: {img_file.name}")

            try:
                # Run inference
                results, inference_time = self.run_inference(img_file, conf, imgsz)

                # Extract detections
                detections = self.process_detections(results)

                # Update statistics
                self.results_stats["total_time"] += inference_time
                self.results_stats["total_detections"] += len(detections)

                if len(detections) > 0:
                    self.results_stats["images_with_detections"] += 1

                # Display results
                print(f"  Time: {inference_time*1000:.2f}ms")
                print(f"  FPS: {1/inference_time:.2f}")
                print(f"  Detections: {len(detections)}")

                for det_idx, detection in enumerate(detections, 1):
                    conf_str = f"{detection['confidence']:.3f}"
                    print(f"    {det_idx}. {detection['class_name']}: {conf_str}")

                # Save annotated image
                self.save_annotated_image(results, img_file)

            except Exception as error:
                logger.error(f"Error processing {img_file.name}: {error}")
                print(f"  Error: {error}")
                continue

        self.results_stats["total_images"] = len(image_files)

        return self.results_stats

    def print_statistics(self) -> None:
        """Display test statistics."""
        print("\n" + "=" * 80)
        print("Test Statistics")
        print("=" * 80)

        total_images = self.results_stats["total_images"]
        total_time = self.results_stats["total_time"]
        total_detections = self.results_stats["total_detections"]
        images_with_det = self.results_stats["images_with_detections"]

        print(f"Total images: {total_images}")
        if total_images > 0:
            pct = images_with_det / total_images * 100
            print(f"Images with detections: {images_with_det} ({pct:.1f}%)")

            avg_time = (total_time / total_images) * 1000
            avg_fps = total_images / total_time
            avg_detections = total_detections / total_images

            print(f"Average inference time: {avg_time:.2f}ms")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total detections: {total_detections}")
            print(f"Average detections per image: {avg_detections:.2f}")

        print(f"\nResults saved to: {self.output_dir}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test YOLOv11 model on image directory"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pt file)",
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing test images",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions",
        help="Output directory for annotated images (default: results/predictions)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Image size for inference (default: 320)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for image testing.

    Tests model on directory of images and generates annotated predictions.
    """
    args = parse_arguments()

    try:
        # Create tester
        tester = ImageTester(
            model_path=args.model, images_dir=args.images, output_dir=args.output
        )

        # Run tests
        tester.test_on_directory(conf=args.conf, imgsz=args.imgsz)

        # Display statistics
        tester.print_statistics()

        print("\n" + "=" * 80)
        print("Testing Completed Successfully")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(1)

    except Exception as error:
        logger.error(f"Testing failed: {error}", exc_info=True)
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
