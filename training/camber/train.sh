#!/usr/bin/env bash
# =============================================================================
# Fine-tuning YOLOv11n на remapped RTSD (Camber Cloud, NVIDIA L4 24GB)
# Запускается из: camber job create --cmd "bash train.sh" --path stash://...
# =============================================================================
set -euo pipefail

echo "=== [$(date '+%H:%M:%S')] Установка зависимостей ==="
pip install --quiet ultralytics==8.3.0 mlflow pyyaml

echo "=== [$(date '+%H:%M:%S')] Проверка GPU ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "=== [$(date '+%H:%M:%S')] Запуск fine-tuning ==="
yolo detect train \
    cfg=training_config.yaml \
    data=rtsd_yolo_belarus/data.yaml \
    model=yolov11n.pt \
    project=runs/finetune \
    name=rtsd_belarus_v1 \
    exist_ok=True

echo "=== [$(date '+%H:%M:%S')] Экспорт в NCNN для Raspberry Pi ==="
BEST_PT=$(find runs/finetune/rtsd_belarus_v1/weights -name "best.pt" | head -1)
if [ -f "$BEST_PT" ]; then
    yolo export model="$BEST_PT" format=ncnn imgsz=320 half=False simplify=True
    echo "NCNN модель сохранена рядом с best.pt"
else
    echo "ВНИМАНИЕ: best.pt не найден, экспорт пропущен"
fi

echo "=== [$(date '+%H:%M:%S')] Обучение завершено ==="
echo "Результаты: runs/finetune/rtsd_belarus_v1/"
ls -lh runs/finetune/rtsd_belarus_v1/weights/ 2>/dev/null || true
