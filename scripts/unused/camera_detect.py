#!/usr/bin/env python3
"""
Детекция знаков с веб-камеры в реальном времени
Для Raspberry Pi и обычных ПК
"""

import argparse
import cv2
import logging
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.6):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        logger.info(f"Загружаю модель: {model_path}")
        self.model = YOLO(model_path)
        logger.info("Модель успешно загружена")

        self.class_names = self.model.names
        self.num_classes = len(self.class_names)

    def detect(self, frame):
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        return results[0]

    def draw_predictions(self, frame, results):
        frame_copy = frame.copy()
        h, w = frame.shape[:2]

        if results.boxes is None or len(results.boxes) == 0:
            return frame_copy

        colors = self._generate_colors(self.num_classes)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names[class_id]

            color = colors[class_id % len(colors)]
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            text = f"{class_name} {confidence:.2%}"
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_size[1] - 4),
                (x1 + text_size[0], y1),
                color,
                -1
            )

            cv2.putText(
                frame_copy,
                text,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return frame_copy

    @staticmethod
    def _generate_colors(num_classes):
        colors = []
        for i in range(num_classes):
            hue = (i * 180 // num_classes) % 180
            hsv = np.uint8([[[hue, 255, 255]]]) # type: ignore
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0] # type: ignore
            colors.append(tuple(map(int, bgr)))
        return colors


def run_webcam(model_path, camera_id=0, conf_threshold=0.25,
               save_output=False, output_dir=None):
    detector = SignDetector(model_path, conf_threshold=conf_threshold)

    logger.info(f"Подключаюсь к камере {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error(f"Не удалось открыть камеру {camera_id}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Камера: {width}x{height} @ {fps} FPS")

    out = None
    if save_output:
        if output_dir is None:
            output_dir = Path("outputs")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"detection_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logger.info(f"Сохраняю видео в {output_path}")

    logger.info("Детекция запущена. Нажмите 'q' для выхода")

    frame_count = 0
    detections_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Ошибка при чтении кадра")
                break

            frame_count += 1
            results = detector.detect(frame)
            frame_with_pred = detector.draw_predictions(frame, results)

            num_detections = len(
                results.boxes) if results.boxes is not None else 0
            detections_count += num_detections

            info_text = f"Frame: {frame_count} | Detections: {num_detections} | Avg: {detections_count/max(1, frame_count):.1f}"
            cv2.putText(
                frame_with_pred,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Sign Detection", frame_with_pred)

            if out is not None:
                out.write(frame_with_pred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Выход...")
                break

    except KeyboardInterrupt:
        logger.info("Прервано пользователем")

    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        logger.info(f"Всего кадров: {frame_count}")
        logger.info(f"Всего детекций: {detections_count}")


def run_image(model_path, image_path, conf_threshold=0.25, output_dir=None):
    detector = SignDetector(model_path, conf_threshold=conf_threshold)

    logger.info(f"Загружаю изображение: {image_path}")
    frame = cv2.imread(image_path)

    if frame is None:
        logger.error(f"Не удалось загрузить изображение: {image_path}")
        return

    logger.info(f"Размер: {frame.shape[1]}x{frame.shape[0]}")

    results = detector.detect(frame)
    frame_with_pred = detector.draw_predictions(frame, results)

    cv2.imshow("Detection Result", frame_with_pred)
    logger.info("Нажмите любую клавишу для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"detection_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(output_path), frame_with_pred)
        logger.info(f"Результат сохранен: {output_path}")

    if results.boxes is not None:
        logger.info(f"Найдено объектов: {len(results.boxes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Детекция дорожных знаков с камеры или изображения"
    )
    parser.add_argument("model", help="Путь к best.pt модели")
    parser.add_argument("--source", type=str, default="0",
                        help="ID камеры или путь к изображению")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Порог уверенности")
    parser.add_argument("--save", action="store_true",
                        help="Сохранять видео/результаты")
    parser.add_argument("--output", type=str,
                        default="outputs", help="Папка для сохранения")

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Модель не найдена: {args.model}")
        return

    if args.source == "0":
        run_webcam(args.model, camera_id=0, conf_threshold=args.conf,
                   save_output=args.save, output_dir=args.output if args.save else None)
    else:
        if not Path(args.source).exists():
            logger.error(f"Изображение не найдено: {args.source}")
            return
        run_image(args.model, args.source, conf_threshold=args.conf,
                  output_dir=args.output if args.save else None)


if __name__ == "__main__":
    main()
