Raspberry Pi

## 1) Что делает

Проект распознаёт дорожные знаки на камере или видео с помощью YOLO

Результат:
- рамки вокруг знаков;
- подпись класса;
- confidence

## 2) Быстрый запуск без Docker (Python runtime)

Требования:
- Python 3.11+;
- Linux/Raspberry Pi;
- модель в [models/deploy](models/deploy).

### 2.1 Установка зависимостей

Из корня проекта:

```bash
python scripts/deploy.py
```

### 2.2 Запуск с камеры

```bash
python scripts/run_raspberry.py \
	--source 0 \
	--output output/raspberry_output.mp4 \
	--model models/deploy/best_ncnn_model \
	--device cpu
```

### 2.3 Запуск с видеофайла

```bash
python scripts/run_raspberry.py \
	--source input.mp4 \
	--output output/result.mp4 \
	--model models/deploy/best_ncnn_model \
	--device cpu
```

Примечание: если NCNN-папки нет, можно передать `--model models/deploy/best.pt`.

## 3) Как запустить Docker-образ на Raspberry Pi

Это отдельный сценарий запуска именно контейнера на малине.

### 3.1 Подготовка на Raspberry Pi

1. Установить Docker.
2. Склонировать проект.
3. Убедиться, что модель лежит в [models/deploy](models/deploy):
	 - либо `best_ncnn_model/`;
	 - либо `best.pt`.

Примечание: экспорт NCNN в этом README намеренно не описывается (внутренний шаг подготовки артефакта).

### 3.2 Сборка образа инференса на Raspberry Pi

Из корня проекта на Raspberry Pi:

```bash
docker build --no-cache --network=host -f docker/Dockerfile.raspberry -t rrs-inference .
```

> **Важно:** флаги `--no-cache` и `--network=host` необходимы для предотвращения проблем с сетевыми таймаутами при скачивании зависимостей на медленных сетях. Сборка займёт 10–15 минут.

### 3.3 Запуск образа с камерой

```bash
docker run --rm -it \
	--device /dev/video0 \
	-v $(pwd)/models/deploy:/models \
	-v $(pwd)/output:/output \
	rrs-inference
```

### 3.4 Запуск образа с видеофайлом

```bash
docker run --rm -it \
	-v $(pwd)/models/deploy:/models \
	-v $(pwd)/output:/output \
	-v /абсолютный/путь/к/video.mp4:/input/video.mp4 \
	rrs-inference --source /input/video.mp4
```

### 3.5 Явно указать модель в контейнере

```bash
docker run --rm -it \
	--device /dev/video0 \
	-v $(pwd)/models/deploy:/models \
	-v $(pwd)/output:/output \
	rrs-inference \
	--source 0 \
	--model /models/best_ncnn_model \
	--device cpu
```

## 4) Типовые ошибки

1. **Ошибка при `docker build` (сетевые таймауты):**
	 - убедись, что используешь флаги `--network=host --no-cache`;
	 - если ошибка повторяется, повтори сборку (pip автоматически повторит загрузку).

2. **Ошибка `THESE PACKAGES DO NOT MATCH THE HASHES` при сборке:**
	 - проверь, не установлена ли переменная `PIP_REQUIRE_HASHES=1` в системе;
	 - сбрось её: `unset PIP_REQUIRE_HASHES`;
	 - повтори сборку.

3. **`Модель не найдена` при запуске контейнера:**
	 - проверь наличие [models/deploy/best_ncnn_model](models/deploy/best_ncnn_model) или [models/deploy/best.pt](models/deploy/best.pt).

4. **Камера не открывается в контейнере:**
	 - проверь `/dev/video*` на хосте: `ls /dev/video*`;
	 - если несколько камер, попробуй другой индекс: `--device /dev/video1`;
	 - убедись, что используешь флаг `--device` в команде `docker run`.

