#!/usr/bin/env bash
# =============================================================================
# Загрузка проекта на Camber Stash и запуск fine-tuning
# Запускать ЛОКАЛЬНО на ноутбуке после установки: pip install camber
# =============================================================================
set -euo pipefail

# ── Настройки — поменяй под свой аккаунт ─────────────────────────────────────
CAMBER_USERNAME="your_camber_username"   # <-- вставь свой логин Camber
PROJECT_NAME="rtsd-belarus-finetune"
STASH_PATH="stash://${CAMBER_USERNAME}/${PROJECT_NAME}"

echo "=== Шаг 0: Авторизация ==="
# Если ещё не залогинена:
# camber login

echo "=== Шаг 1: Создание папки в Stash ==="
camber stash mkdir "${STASH_PATH}" 2>/dev/null || true

echo "=== Шаг 2: Загрузка скриптов обучения ==="
camber stash cp train.sh           "${STASH_PATH}/train.sh"
camber stash cp training_config.yaml "${STASH_PATH}/training_config.yaml"
camber stash cp belarus_mapping_data.json "${STASH_PATH}/belarus_mapping_data.json"

echo "=== Шаг 3: Загрузка remapped датасета ==="
echo "  Это займёт время (датасет большой) — запускай отдельно если нужно"
camber stash cp -r rtsd_yolo_belarus/ "${STASH_PATH}/rtsd_yolo_belarus/"

echo "=== Шаг 4: Проверка загрузки ==="
camber stash ls "${STASH_PATH}/"

echo "=== Шаг 5: Запуск обучения на GPU ==="
camber job create \
    --cmd "bash train.sh" \
    --path "${STASH_PATH}" \
    --size SMALL \
    --engine mpi \
    --tag "yolo-finetune" \
    --tag "rtsd-belarus"

echo ""
echo "✅ Job запущен!"
echo "   Следить за статусом: camber job list"
echo "   Логи в реальном времени: camber job log <JOB_ID> ."
echo "   Скачать результаты: camber stash cp -r '${STASH_PATH}/runs/' ./runs/"
