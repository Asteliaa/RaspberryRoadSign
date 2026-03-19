#!/usr/bin/env python3
"""Универсальный запуск инференса для Raspberry Pi (файл или камера)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def resolve_default_model() -> Path:
    """Вернуть модель по умолчанию для Raspberry runtime.

    Приоритет:
    1) NCNN-папка после экспорта (быстрее на RPi)
    2) Локальные PT-веса
    3) Архивный fallback
    """
    ncnn_dir = PROJECT_ROOT / "models" / "deploy" / "best_ncnn_model"
    ncnn_param = ncnn_dir / "model.ncnn.param"
    ncnn_bin = ncnn_dir / "model.ncnn.bin"
    if ncnn_dir.is_dir() and ncnn_param.exists() and ncnn_bin.exists():
        return ncnn_dir

    pt_model = PROJECT_ROOT / "models" / "deploy" / "best.pt"
    if pt_model.exists():
        return pt_model

    return (
        PROJECT_ROOT
        / "_archive"
        / "rebuild_2026_03_15"
        / "runs"
        / "rtsd_train"
        / "rtsd_yolo11n_pi_other50"
        / "weights"
        / "best.pt"
    )


DEFAULT_MODEL = resolve_default_model()


def parse_source(source: str) -> int | Path:
    """Преобразовать источник в индекс камеры или путь к видео."""
    if source.isdigit():
        return int(source)
    return Path(source)


def run_camera(
    detector: Any,
    source_idx: int,
    output_path: Path,
    show: bool,
) -> None:
    """Инференс с USB/RPi камеры."""
    import cv2
    from RaspberryRoadSign.inference.visualizer import Visualizer

    cap = cv2.VideoCapture(source_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру: {source_idx}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать выходной файл: {output_path}")

    logger = logging.getLogger("raspberry-main")
    visualizer = Visualizer()
    frame_count = 0
    detections = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            results = detector.detect_frame(frame)

            if results and results.boxes:
                detections += len(results.boxes)
                frame = detector._annotate_frame(frame, results, visualizer)

            writer.write(frame)

            if show:
                cv2.imshow("RaspberryRoadSign", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 30 == 0:
                logger.info("Кадров: %s, детекций: %s", frame_count, detections)

    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    logger.info("Готово. Сохранено: %s", output_path)
    logger.info("Итог: кадров=%s, детекций=%s", frame_count, detections)


def main() -> None:
    parser = argparse.ArgumentParser(description="RaspberryRoadSign inference")
    parser.add_argument(
        "--source",
        default="0",
        help="Источник: индекс камеры (например 0) или путь к видео",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Путь к весам модели",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "raspberry_output.mp4",
        help="Путь для выходного видео",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Порог confidence")
    parser.add_argument(
        "--device",
        default="auto",
        help="Устройство: auto/cpu/cuda",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать окно предпросмотра (Q для выхода)",
    )
    parser.add_argument("--log-level", default="INFO", help="Уровень логов")
    args = parser.parse_args()

    try:
        from RaspberryRoadSign.inference.detector import TrafficSignDetector
        from RaspberryRoadSign.utils.logging import setup_logging
    except ModuleNotFoundError as exc:
        missing = str(exc).split("No module named ")[-1].strip("'")
        raise SystemExit(
            "Отсутствует зависимость: "
            f"{missing}. Установите пакеты: pip install -r requirements_raspberry.txt"
        ) from exc

    setup_logging(log_level=args.log_level, name="raspberry-main")
    logger = logging.getLogger("raspberry-main")

    if not args.model.exists():
        raise FileNotFoundError(f"Модель не найдена: {args.model}")

    detector = TrafficSignDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )

    source = parse_source(args.source)
    if isinstance(source, int):
        run_camera(detector, source, args.output, args.show)
        return

    if not source.exists():
        raise FileNotFoundError(f"Видео не найдено: {source}")

    stats = detector.detect_video(source, args.output)
    logger.info("Готово. Детекций: %s, кадров: %s", stats["detections"], stats["total_frames"])


if __name__ == "__main__":
    main()