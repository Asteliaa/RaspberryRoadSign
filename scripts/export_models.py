#!/usr/bin/env python3
"""
Export trained YOLOv11 model to multiple formats.

Exports model to ONNX, TensorFlow Lite, and TorchScript formats
optimized for Raspberry Pi 4B edge device inference.
"""

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import onnx
import onnxruntime as ort
from ultralytics import YOLO


logger = logging.getLogger(__name__)

# Export configuration
EXPORT_FORMATS = {
    "onnx": {"opset": 12, "simplify": True},
    "tflite": {"int8": False},
    "torchscript": {},
}


class ModelExporter:
    """
    Export YOLOv11 model for Raspberry Pi deployment.

    Supports ONNX (recommended), TensorFlow Lite, and TorchScript formats.
    Verifies exports and saves metadata.
    """

    def __init__(self, model_path: Path | str, output_dir: Path | str = "exports"):
        """
        Initialize model exporter.

        Args:
            model_path: Path to trained model (.pt file).
            output_dir: Output directory for exports.

        Raises:
            FileNotFoundError: If model file not found.
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        print("=" * 80)
        print("Model Export for Raspberry Pi 4B")
        print("=" * 80)
        print(f"Model: {self.model_path}")
        print(f"Output directory: {self.output_dir}")

        self.model = YOLO(str(self.model_path))

        # Display model size
        model_size_mb = self.model_path.stat().st_size / (1024**2)
        print(f"Model size: {model_size_mb:.2f} MB")

    def export_onnx(
        self,
        imgsz: int = 320,
        simplify: bool = True,
        dynamic: bool = False,
        opset: int = 12,
    ) -> Optional[Path]:
        """
        Export model to ONNX format (recommended for Raspberry Pi).

        ONNX Runtime provides CPU-optimized inference on edge devices.

        Args:
            imgsz: Input image size.
            simplify: Simplify ONNX graph.
            dynamic: Use dynamic batch size (not recommended for Pi).
            opset: ONNX operator set version.

        Returns:
            Path to exported ONNX model if successful, None otherwise.
        """
        print("\n" + "=" * 80)
        print("Exporting to ONNX Format")
        print("=" * 80)

        start_time = time.time()

        try:
            # Export to ONNX
            onnx_path = self.model.export(
                format="onnx",
                imgsz=imgsz,
                simplify=simplify,
                dynamic=dynamic,
                opset=opset,
                half=False,  # FP32 for CPU
            )

            export_time = time.time() - start_time

            # Verify export
            onnx_file = Path(onnx_path)
            if not onnx_file.exists():
                logger.error("ONNX file not created")
                return None

            size_mb = onnx_file.stat().st_size / (1024**2)

            print(f"\nONNX Export Successful")
            print(f"  File: {onnx_file}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Export time: {export_time:.2f}s")

            # Copy to export directory
            dest = self.output_dir / "onnx" / onnx_file.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            if onnx_file != dest:
                shutil.copy(str(onnx_file), str(dest))
                print(f"  Saved to: {dest}")

            # Verify ONNX model
            self._verify_onnx(dest)

            return dest

        except Exception as error:
            logger.error(f"ONNX export failed: {error}")
            print(f"Error: {error}")
            traceback.print_exc()
            return None

    def _verify_onnx(self, onnx_path: Path) -> bool:
        """
        Verify ONNX model integrity.

        Args:
            onnx_path: Path to ONNX model file.

        Returns:
            True if verification successful, False otherwise.
        """
        print("\nVerifying ONNX Model...")

        try:
            # Load and check model structure
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            print("  Structure: Valid")

            # Test with ONNX Runtime
            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )

            # Get input/output information
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]

            print("  ONNX Runtime: Ready")
            print(f"  Input: {input_info.name} {input_info.shape}")
            print(f"  Output: {output_info.name} {output_info.shape}")

            return True

        except Exception as error:
            logger.warning(f"ONNX verification failed: {error}")
            return False

    def export_tflite(self, imgsz: int = 320, int8: bool = False) -> Optional[Path]:
        """
        Export model to TensorFlow Lite format.

        Provides alternative CPU inference option for Raspberry Pi.

        Args:
            imgsz: Input image size.
            int8: Use INT8 quantization (reduces size and speeds up).

        Returns:
            Path to exported TFLite model if successful, None otherwise.
        """
        print("\n" + "=" * 80)
        print("Exporting to TensorFlow Lite Format")
        print("=" * 80)

        start_time = time.time()

        try:
            # Export to TFLite
            tflite_path = self.model.export(format="tflite", imgsz=imgsz, int8=int8)

            export_time = time.time() - start_time

            # Verify export
            tflite_file = Path(tflite_path)
            if not tflite_file.exists():
                logger.error("TFLite file not created")
                return None

            size_mb = tflite_file.stat().st_size / (1024**2)

            print(f"\nTFLite Export Successful")
            print(f"  File: {tflite_file}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Quantization (INT8): {int8}")
            print(f"  Export time: {export_time:.2f}s")

            # Copy to export directory
            dest = self.output_dir / "tflite" / tflite_file.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            if tflite_file != dest:
                shutil.copy(str(tflite_file), str(dest))
                print(f"  Saved to: {dest}")

            return dest

        except Exception as error:
            logger.error(f"TFLite export failed: {error}")
            print(f"Error: {error}")
            return None

    def export_torchscript(self, imgsz: int = 320) -> Optional[Path]:
        """
        Export model to TorchScript format.

        Supports PyTorch runtime on Raspberry Pi.

        Args:
            imgsz: Input image size.

        Returns:
            Path to exported TorchScript model if successful, None otherwise.
        """
        print("\n" + "=" * 80)
        print("Exporting to TorchScript Format")
        print("=" * 80)

        start_time = time.time()

        try:
            # Export to TorchScript
            torchscript_path = self.model.export(format="torchscript", imgsz=imgsz)

            export_time = time.time() - start_time

            # Verify export
            ts_file = Path(torchscript_path)
            if not ts_file.exists():
                logger.error("TorchScript file not created")
                return None

            size_mb = ts_file.stat().st_size / (1024**2)

            print(f"\nTorchScript Export Successful")
            print(f"  File: {ts_file}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Export time: {export_time:.2f}s")

            # Copy to export directory
            dest = self.output_dir / "torchscript" / ts_file.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            if ts_file != dest:
                shutil.copy(str(ts_file), str(dest))
                print(f"  Saved to: {dest}")

            return dest

        except Exception as error:
            logger.error(f"TorchScript export failed: {error}")
            print(f"Error: {error}")
            return None

    def export_all(self, imgsz: int = 320) -> Dict[str, Optional[Path]]:
        """
        Export model to all supported formats.

        Args:
            imgsz: Input image size.

        Returns:
            Dictionary mapping format names to exported model paths.
        """
        results = {}

        # ONNX (recommended for Pi)
        results["onnx"] = self.export_onnx(imgsz=imgsz, simplify=True)

        # TFLite variants
        results["tflite"] = self.export_tflite(imgsz=imgsz, int8=False)
        results["tflite_int8"] = self.export_tflite(imgsz=imgsz, int8=True)

        # TorchScript
        results["torchscript"] = self.export_torchscript(imgsz=imgsz)

        # Save export information
        self._save_export_info(results, imgsz)

        return results

    def _save_export_info(self, results: Dict[str, Optional[Path]], imgsz: int) -> None:
        """
        Save export metadata to JSON file.

        Args:
            results: Dictionary of export results.
            imgsz: Image size used for exports.
        """
        info = {
            "source_model": str(self.model_path),
            "image_size": imgsz,
            "exports": {},
        }

        for format_name, export_path in results.items():
            if export_path and export_path.exists():
                size_mb = export_path.stat().st_size / (1024**2)
                info["exports"][format_name] = {
                    "path": str(export_path),
                    "size_mb": round(size_mb, 2),
                }

        info_file = self.output_dir / "export_info.json"
        with open(info_file, "w") as file:
            json.dump(info, file, indent=2)

        print(f"\nMetadata saved: {info_file}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export YOLOv11 model for Raspberry Pi"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.pt file)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="exports",
        help="Output directory for exports (default: exports)",
    )

    parser.add_argument(
        "--imgsz", type=int, default=320, help="Image size for export (default: 320)"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["all", "onnx", "tflite", "torchscript"],
        help="Export format (default: all)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for model export.

    Orchestrates export to specified formats and generates metadata.
    """
    args = parse_arguments()

    try:
        # Create exporter
        exporter = ModelExporter(model_path=args.model, output_dir=args.output)

        # Execute export
        if args.format == "all":
            results = exporter.export_all(imgsz=args.imgsz)
        elif args.format == "onnx":
            results = {"onnx": exporter.export_onnx(imgsz=args.imgsz)}
        elif args.format == "tflite":
            results = {"tflite": exporter.export_tflite(imgsz=args.imgsz)}
        elif args.format == "torchscript":
            results = {"torchscript": exporter.export_torchscript(imgsz=args.imgsz)}

        # Display results
        print("\n" + "=" * 80)
        print("Export Completed Successfully")
        print("=" * 80)

        for format_name, export_path in results.items():
            if export_path and export_path.exists():
                size_mb = export_path.stat().st_size / (1024**2)
                print(f"  {format_name:15s}: {size_mb:6.2f} MB")
                print(f"    Path: {export_path}")

        print("\n" + "=" * 80)
        print("Next Steps for Raspberry Pi Deployment:")
        print("=" * 80)
        print("  1. Transfer ONNX model to Raspberry Pi")
        print("  2. Install ONNX Runtime: pip install onnxruntime")
        print("  3. Run inference: python scripts/12_webcam_detection.py")

    except KeyboardInterrupt:
        print("\nExport interrupted by user")
        sys.exit(1)

    except Exception as error:
        logger.error(f"Export failed: {error}", exc_info=True)
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
