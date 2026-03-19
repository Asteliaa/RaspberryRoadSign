#!/usr/bin/env python3
"""
Экспорт best.pt → NCNN формат для деплоя на Raspberry Pi.

Запускается внутри Docker-контейнера rrs-export.
Входной файл:  /models/best.pt
Выходная папка: /output/best_ncnn_model/
"""

import shutil
import sys
from pathlib import Path

MODEL_IN  = Path("/models/best.pt")
OUTPUT_DIR = Path("/output")


def main() -> None:
    # Проверяем входной файл
    if not MODEL_IN.exists():
        print(f"[ERROR] Модель не найдена: {MODEL_IN}")
        print("  Скрипт нужно запускать внутри Docker-контейнера rrs-export.")
        print("  Пример запуска из корня проекта:")
        print("  docker run --rm -v $(pwd)/models/deploy:/models -v $(pwd)/output_ncnn:/output rrs-export")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Загружаем модель: {MODEL_IN}")
    from ultralytics import YOLO

    model = YOLO(str(MODEL_IN))

    # Показываем информацию о модели
    print(f"[INFO] Task:   {model.task}")
    try:
        n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        print(f"[INFO] Params: {n_params:.2f}M")
    except Exception:
        pass

    print("[INFO] Экспортируем в NCNN (imgsz=320, half=False)...")
    print("       imgsz=320 — оптимально для RPi 4 (~30ms inference)")
    print("       half=False — RPi 4 CPU не поддерживает FP16")

    # Экспорт происходит рядом с исходным файлом, потом перемещаем
    export_path = model.export(
        format="ncnn",
        imgsz=320,
        half=False,
        batch=1,
        simplify=True,   # упрощает ONNX граф перед конвертацией
        device="cpu",
    )

    exported = Path(export_path)
    dest = OUTPUT_DIR / exported.name

    # Если папка уже существует — удаляем старую версию
    if dest.exists():
        shutil.rmtree(dest)

    shutil.move(str(exported), str(dest))

    print(f"\n[OK] Экспорт завершён: {dest}")
    print(f"     Файлы внутри:")
    for f in sorted(dest.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"       {f.name:<35} {size_kb:.1f} KB")

    print(f"\n[NEXT] Скопируй папку на Raspberry Pi:")
    print(f"  scp -r ./output_ncnn/best_ncnn_model pi@<IP>:~/RaspberryRoadSign/models/deploy/")


if __name__ == "__main__":
    main()
