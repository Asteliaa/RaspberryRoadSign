#!/usr/bin/env python3
"""
Real-time traffic sign detection from webcam.

Detects Belarusian/European traffic signs using YOLOv11 model
with real-time visualization and performance monitoring.
"""

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===== EDIT THESE PARAMETERS =====
# Model configuration
DEFAULT_MODEL_PATH = "runs/rtsd_train/rtsd_yolo11m_v1/weights/best.pt"
DEFAULT_CAMERA_ID = 0
DEFAULT_CONF_THRESHOLD = 0.5

# Camera configuration - 640x480 (что вернула камера)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Display configuration - УВЕЛИЧЕННОЕ ОКНО
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 960
WINDOW_NAME = "🚦 Belarusian Traffic Sign Detection"

# UI Configuration
INFO_PANEL_HEIGHT = 120
FPS_QUEUE_SIZE = 30

# Color thresholds for confidence visualization
HIGH_CONFIDENCE_THRESHOLD = 0.7
MEDIUM_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_COLOR = (0, 255, 0)  # Green
MEDIUM_CONFIDENCE_COLOR = (0, 255, 255)  # Yellow
LOW_CONFIDENCE_COLOR = (0, 165, 255)  # Orange
# ===== END CONFIGURATION =====

# Import class mapping
try:
    from class_mapping import GTSRB_CLASSES, READABLE_NAMES
except ImportError:
    READABLE_NAMES = {i: f"class_{i}" for i in range(43)}
    GTSRB_CLASSES = READABLE_NAMES


class TrafficSignDetector:
    """
    Real-time traffic sign detector from webcam feed.

    Detects Belarusian/European traffic signs, displays results,
    and provides interactive controls for threshold adjustment.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        camera_id: int = DEFAULT_CAMERA_ID,
    ):
        """
        Initialize detector.

        Args:
            model_path: Path to trained model (.pt or .onnx).
            conf_threshold: Initial confidence threshold.
            camera_id: Webcam device ID (default 0 for built-in).

        Raises:
            RuntimeError: If camera cannot be opened.
            FileNotFoundError: If model file not found.
        """
        print("=" * 80)
        print("🚦 Belarusian Traffic Sign Detection - Real-time")
        print("=" * 80)
        print("Dataset: GTSRB-based (43 European traffic sign classes)\n")

        # Load model
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"❌ Model not found: {model_path}")

        print(f"📦 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print("✅ Model loaded successfully!\n")

        # Initialize camera
        print(f"📷 Initializing camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Failed to open camera {camera_id}")

        # Установи разрешение
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get actual resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print("✅ Camera initialized successfully!")
        print(f"   Capture Resolution: {width}x{height}")
        print(f"   Display Resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
        print(f"   FPS: {fps}\n")

        # Resize окна
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

        # Performance tracking
        self.fps_queue: deque = deque(maxlen=FPS_QUEUE_SIZE)

        # Statistics
        self.frame_count = 0
        self.detection_count = 0

        # Display configuration
        print("⚙️  Configuration:")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   Model: {model_path}\n")

        print("⌨️  Keyboard Controls:")
        print("   Q / ESC    - Exit")
        print("   S          - Save frame")
        print("   SPACE      - Pause/Resume")
        print("   + / -      - Adjust confidence threshold\n")
        print("=" * 80)
        print("▶️  Starting Real-time Detection - Press Q to quit")
        print("=" * 80 + "\n")

    def draw_info_panel(
        self, frame: np.ndarray, fps: float, detections: List[Dict]
    ) -> np.ndarray:
        """
        Draw information panel on frame.

        Args:
            frame: Input frame.
            fps: Current frames per second.
            detections: List of detections in current frame.

        Returns:
            Frame with info panel drawn.
        """
        # Semi-transparent overlay panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (frame.shape[1], INFO_PANEL_HEIGHT), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Information lines
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Detections: {len(detections)}",
            f"Conf: {self.conf_threshold:.2f}",
            "🚦 Belarusian Traffic Signs",
        ]

        y_position = 25
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_position += 22

        return frame

    def draw_detections(
        self, frame: np.ndarray, results: List
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Draw detection boxes and labels on frame.

        Args:
            frame: Input frame.
            results: YOLO detection results.

        Returns:
            Tuple of (annotated_frame, detections_list).
        """
        detections: List[Dict] = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Extract box data
                coordinates = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coordinates
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                if conf < self.conf_threshold:
                    continue

                # Get class names
                readable_name = READABLE_NAMES.get(class_id, f"class_{class_id}")
                full_name = GTSRB_CLASSES.get(class_id, readable_name)
                detections.append(
                    {
                        "class_name": readable_name,
                        "confidence": conf,
                        "full_name": full_name,
                    }
                )

                # Determine box color based on confidence
                if conf > HIGH_CONFIDENCE_THRESHOLD:
                    box_color = HIGH_CONFIDENCE_COLOR
                elif conf > MEDIUM_CONFIDENCE_THRESHOLD:
                    box_color = MEDIUM_CONFIDENCE_COLOR
                else:
                    box_color = LOW_CONFIDENCE_COLOR

                # Draw bounding box
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2
                )

                # Prepare label text
                label_text = f"{readable_name}: {conf:.2f}"

                # Get text size for background
                text_size, baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                text_width, text_height = text_size

                # Draw label background
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1) - text_height - baseline - 5),
                    (int(x1) + text_width, int(y1)),
                    box_color,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    frame,
                    label_text,
                    (int(x1), int(y1) - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

                # Draw full name below box
                cv2.putText(
                    frame,
                    full_name,
                    (int(x1), int(y2) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                )

        return frame, detections

    def save_frame(self, frame: np.ndarray) -> None:
        """
        Save current frame to file.

        Args:
            frame: Frame to save.
        """
        filename = f"detection_frame_{self.frame_count}.jpg"
        success = cv2.imwrite(filename, frame)

        if success:
            print(f"💾 Frame saved: {filename}")
        else:
            logger.error(f"Failed to save frame: {filename}")

    def adjust_confidence(self, delta: float) -> None:
        """
        Adjust confidence threshold.

        Args:
            delta: Change amount (positive or negative).
        """
        self.conf_threshold = np.clip(self.conf_threshold + delta, 0.0, 1.0)
        print(f"🎯 Confidence threshold: {self.conf_threshold:.2f}")

    def run(self) -> None:
        """
        Main detection loop with interactive controls.

        Continuously captures frames, runs inference,
        and handles keyboard input.
        """
        paused = False
        frame = None

        try:
            while True:
                if not paused:
                    # Read frame with error handling
                    ret, frame = self.cap.read()

                    if not ret or frame is None:
                        print("⚠️  Failed to read frame from camera")
                        time.sleep(0.1)
                        continue

                    self.frame_count += 1

                    # Measure inference time
                    start_time = time.time()
                    results = self.model(frame, conf=0.1, verbose=False)
                    inference_time = time.time() - start_time

                    # Calculate FPS
                    current_fps = 1.0 / (inference_time + 0.001)
                    self.fps_queue.append(current_fps)
                    avg_fps = np.mean(self.fps_queue) if self.fps_queue else 0

                    # Draw detections
                    frame, detections = self.draw_detections(frame, results)

                    # Draw info panel
                    frame = self.draw_info_panel(frame, avg_fps, detections)

                    # Update statistics
                    if detections:
                        self.detection_count += len(detections)

                # Display frame (OpenCV автоматически масштабирует)
                if frame is not None:
                    cv2.imshow(WINDOW_NAME, frame)
                else:
                    print("⚠️  No frame to display")
                    time.sleep(0.1)
                    continue

                # Handle keyboard input (не блокирующий - 30ms timeout)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q") or key == 27:  # Q or ESC
                    print("\n✅ Exiting...")
                    break
                elif key == ord("s"):  # Save frame
                    self.save_frame(frame)
                elif key == ord(" "):  # Pause/Resume
                    paused = not paused
                    status = "⏸️  Paused" if paused else "▶️  Resumed"
                    print(status)
                elif key == ord("+") or key == ord("="):  # Increase threshold
                    self.adjust_confidence(0.05)
                elif key == ord("-"):  # Decrease threshold
                    self.adjust_confidence(-0.05)

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")

        finally:
            # Print session statistics
            self._print_statistics()

            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()

            print("\n✅ Detection completed\n")

    def _print_statistics(self) -> None:
        """Print session statistics."""
        print("\n" + "=" * 80)
        print("📊 Session Statistics")
        print("=" * 80)
        print(f"Total frames:         {self.frame_count}")
        print(f"Total detections:     {self.detection_count}")

        if self.frame_count > 0:
            if self.fps_queue:
                avg_fps = np.mean(self.fps_queue)
            else:
                avg_fps = 0
            detections_per_frame = self.detection_count / self.frame_count

            print(f"Average FPS:          {avg_fps:.2f}")
            print(f"Detections per frame: {detections_per_frame:.2f}")

        print("=" * 80)


def list_available_cameras() -> List[int]:
    """
    List all available cameras.

    Returns:
        List of available camera IDs.
    """
    print("🔍 Scanning for available cameras...\n")

    available_cameras: List[int] = []

    for camera_id in range(10):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            available_cameras.append(camera_id)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  ✅ Camera {camera_id}: {width}x{height}")
            cap.release()

    if not available_cameras:
        print("  ❌ No cameras found\n")
    else:
        print(f"\n✅ Found {len(available_cameras)} camera(s)")
        print("Use --camera <ID> to select\n")

    return available_cameras


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Real-time Belarusian traffic sign detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with default model
  python 12_webcam_detection.py

  # Custom model
  python 12_webcam_detection.py --model runs/rtsd_train/rtsd_yolo11m_v1/weights/best.pt

  # With specific camera and confidence
  python 12_webcam_detection.py --model weights/best.pt --camera 0 --conf 0.6

  # List available cameras
  python 12_webcam_detection.py --list-cameras
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model (.pt or .onnx) (default: {DEFAULT_MODEL_PATH})",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_ID,
        help=f"Webcam device ID (default: {DEFAULT_CAMERA_ID})",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESHOLD})",
    )

    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for webcam detection.

    Initializes detector and runs real-time detection loop.
    """
    args = parse_arguments()

    # List cameras if requested
    if args.list_cameras:
        list_available_cameras()
        return

    try:
        # Create detector
        detector = TrafficSignDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            camera_id=args.camera,
        )

        # Run detection
        detector.run()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user\n")
        sys.exit(0)
    except FileNotFoundError as error:
        print(f"\n❌ File error: {error}\n")
        sys.exit(1)
    except RuntimeError as error:
        print(f"\n❌ Runtime error: {error}\n")
        sys.exit(1)
    except Exception as error:
        logger.error(f"❌ Error: {error}", exc_info=True)
        print(f"\n❌ Error: {error}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
