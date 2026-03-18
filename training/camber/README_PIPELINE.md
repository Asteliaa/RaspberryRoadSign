# RTSD → Belarus Fine-tuning Pipeline

## Структура файлов

```
camber_finetune/
├── remap_rtsd.py              # Конвертация COCO JSON → YOLO + ремаппинг классов
├── training_config.yaml       # Конфиг обучения (fine-tuning, L4 GPU)
├── train.sh                   # Скрипт для Camber (устанавливает deps + запускает yolo)
├── upload_to_camber.sh        # Загрузка на Camber Stash + запуск job
├── belgium_mapping_data.json  # Маппинг данных (RTSD_ID → BY_code, BY_code → new_id)
└── class_mapping_updated.py   # Обновлённый class_mapping.py для репозитория
```

## Шаг 1 — Ремаппинг датасета (локально)

```bash
# Поставь зависимости
pip install tqdm

# Запусти конвертацию
python remap_rtsd.py \
    --coco_dir /path/to/rtsd_coco \
    --output_dir rtsd_yolo_belarus \
    --mapping_json belarus_mapping_data.json

# Результат:
# rtsd_yolo_belarus/
# ├── train/images/  + train/labels/
# ├── val/images/    + val/labels/
# ├── test/images/   + test/labels/
# └── data.yaml      (140 классов белорусских знаков)
```

**Ожидаемая структура --coco_dir:**
```
rtsd_coco/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

## Шаг 2 — Установка Camber CLI

```bash
pip install camber
camber login
```

## Шаг 3 — Загрузка и запуск

```bash
# Отредактируй CAMBER_USERNAME в upload_to_camber.sh
nano upload_to_camber.sh

# Запусти загрузку и обучение
bash upload_to_camber.sh
```

## Шаг 4 — Мониторинг

```bash
camber job list                          # статус всех jobs
camber job get <JOB_ID>                  # детали конкретного job
camber job log <JOB_ID> ./logs/          # скачать логи
```

## Шаг 5 — Скачать результаты

```bash
STASH="stash://your_username/rtsd-belarus-finetune"

# Скачать лучшие веса
camber stash cp "${STASH}/runs/finetune/rtsd_belarus_v1/weights/best.pt" ./models/deploy/

# Скачать NCNN модель для Raspberry Pi
camber stash cp -r "${STASH}/runs/finetune/rtsd_belarus_v1/weights/best_ncnn_model/" ./models/deploy/
```

## Шаг 6 — Обновить репозиторий

Скопировать `class_mapping_updated.py` → `src/RaspberryRoadSign/utils/class_mapping.py`

```bash
cp class_mapping_updated.py ../src/RaspberryRoadSign/utils/class_mapping.py
```

## Что получится

- **140 белорусских классов** (вместо 155 RTSD-шных, из которых 10 без эквивалента)
- **YOLOv11n fine-tuned** на правильных метках
- **NCNN модель** готовая для Raspberry Pi (imgsz=320)
- **Обновлённый class_mapping.py** — корректные подписи в live-inference
