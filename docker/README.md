# Docker: Экспорт и деплой на Raspberry Pi

Два контейнера — два шага:

| Контейнер | Где запускать | Что делает |
|---|---|---|
| `Dockerfile.export` | Ноутбук / Camber (x86) | `best.pt` → `best_ncnn_model/` |
| `Dockerfile.raspberry` | Raspberry Pi 4B (ARM64) | Инференс с камеры / видео |

---

## Шаг 1 — Экспорт модели (на ноутбуке/x86)

> Экспортировать нужно на x86, не на малине. На ARM NCNN экспорт нестабилен.

### 1.1 Собери образ

```bash
# Из корня репозитория
docker build -f docker/Dockerfile.export -t rrs-export .
```

### 1.2 Запусти экспорт

```bash
docker run --rm \
  -v $(pwd)/models/deploy:/models \
  -v $(pwd)/output_ncnn:/output \
  rrs-export
```

Что произойдёт:
- Загружается `./models/deploy/best.pt`
- Экспортируется в NCNN с `imgsz=320` (оптимально для RPi 4)
- Результат сохраняется в `./output_ncnn/best_ncnn_model/`

Внутри `best_ncnn_model/` будут два файла:
```
best_ncnn_model/
  ├── model.ncnn.param   # архитектура (~15 KB)
  └── model.ncnn.bin     # веса (~5-6 MB)
```

### 1.3 Скопируй модель на малину

```bash
# Замени <RASPBERRY_IP> на IP-адрес малины
scp -r ./output_ncnn/best_ncnn_model \
    pi@<RASPBERRY_IP>:~/RaspberryRoadSign/models/deploy/
```

Проверить IP малины: `hostname -I` (выполнить на малине).

---

## Шаг 2 — Запуск на Raspberry Pi 4B

### 2.1 Установи Docker на малину (один раз)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Перелогинься или выполни:
newgrp docker
```

### 2.2 Клонируй репозиторий на малину

```bash
git clone https://github.com/Asteliaa/RaspberryRoadSign.git
cd RaspberryRoadSign
```

Если уже клонирован — просто обнови:
```bash
cd ~/RaspberryRoadSign && git pull
```

### 2.3 Собери образ (на малине)

```bash
# Сборка займёт ~5-10 минут (скачивает wheel для ARM)
docker build -f docker/Dockerfile.raspberry -t rrs-inference .
```

> **Важно:** образ собирается прямо на малине (`arm64v8/python:3.11-slim`).
> Не пытайся перенести образ с x86 — архитектура не совпадёт.

### 2.4 Запусти инференс

**С USB-камеры или RPi Camera Module:**
```bash
docker run --rm -it \
  --device /dev/video0 \
  -v $(pwd)/models/deploy:/models \
  -v $(pwd)/output:/output \
  rrs-inference
```

**С видеофайла:**
```bash
docker run --rm -it \
  -v $(pwd)/models/deploy:/models \
  -v $(pwd)/output:/output \
  -v /путь/к/видео.mp4:/input/video.mp4 \
  rrs-inference --source /input/video.mp4
```

**Указать другую модель явно:**
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

---

## Ожидаемая производительность на RPi 4B

| Формат | imgsz | Время инференса | FPS |
|---|---|---|---|
| `.pt` (PyTorch) | 640 | ~800ms | ~1 |
| `.pt` (PyTorch) | 320 | ~300ms | ~3 |
| **NCNN** | **320** | **~30-50ms** | **~20-30** |
| ONNX | 320 | ~120ms | ~8 |

NCNN — единственный формат, дающий реальный реалтайм на RPi 4B без акселератора.

---

## Диагностика

### Камера не видна в контейнере

```bash
# Проверь что камера определилась в системе
ls /dev/video*
v4l2-ctl --list-devices

# Если несколько камер — попробуй /dev/video1
docker run --rm -it --device /dev/video1 ...
```

### Ошибка "model not found"

Убедись что путь к модели правильный:
```bash
ls $(pwd)/models/deploy/best_ncnn_model/
# Должно быть: model.ncnn.param  model.ncnn.bin
```

### Низкий FPS

- Проверь что модель — именно NCNN, не `.pt`
- `imgsz=320` должен быть указан при экспорте (уже сделано по умолчанию)
- Убедись что малина не троттлит от перегрева: `vcgencmd measure_temp`
- Рекомендуется радиатор + вентилятор

### Пересборка образа после изменения кода

```bash
docker build --no-cache -f docker/Dockerfile.raspberry -t rrs-inference .
```

---

## Структура файлов

```
docker/
  Dockerfile.export      # x86: best.pt → NCNN
  Dockerfile.raspberry   # ARM64: инференс на RPi
  export_ncnn.py         # скрипт экспорта (запускается внутри Dockerfile.export)
  README.md              # этот файл
```
