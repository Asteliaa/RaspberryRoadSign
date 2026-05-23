# Road Sign Detector on Raspberry Pi

> OpenCV DNN-based road sign detection system in a Docker container with USB camera streaming.


### Зачистка хост-системы (выполнять **вне** Docker)

**1. Остановите и полностью отключите службу-перехватчик:**

```bash
sudo systemctl stop motion
sudo systemctl disable motion
```

**2. Проверьте, не держит ли кто-то `/dev/video0`:**

```bash
sudo fuser -v /dev/video0
```

**3. Если команда вывела PID какого-то процесса** (например, `motion` или остаточный `python3`) — принудительно убейте его:

```bash
sudo fuser -k /dev/video0
```

**4. Проверьте готовность камеры (Linux-тест железа):**

```bash
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=1 --stream-to=test.jpg
```

Если команда выполнилась успешно и файл `test.jpg` создался — камера **свободна и готова к работе** с Docker.

---

## Сборка и запуск контейнера

Выполняйте следующие команды в терминале в папке `~/app` на Raspberry Pi.

### Шаг 1: Сборка Docker-образа

```bash
docker build -t rpi-detector .
```

> Docker соберёт образ примерно за 1 минуту благодаря лёгкому базовому образу.

### Шаг 2: Запуск в «привилегированном режиме»

Для обхода ограничений виртуализации Docker на чтение USB-портов и подсистемы `udev` используются флаги `--privileged` и монтирование `-v /dev:/dev`:

```bash
# Удаляем старый контейнер, если он существовал
docker stop road_sign_app 2>/dev/null && docker rm road_sign_app 2>/dev/null

# Запуск свежей стабильной сборки
docker run -d \
  --name road_sign_app \
  -p 5000:5000 \
  --privileged \
  -v /dev:/dev \
  --restart unless-stopped \
  rpi-detector
```

> Флаг `--restart unless-stopped` гарантирует, что при перезапуске Raspberry Pi (например, после отключения питания) детектор знаков поднимется **автоматически**, без необходимости SSH-подключения.

---

## Подключение к веб-интерфейсу

Убедитесь, что Raspberry Pi и ваше принимающее устройство (ноутбук, планшет или телефон) **подключены к одной Wi-Fi сети** (например, к мобильной точке доступа).

**1. Узнайте IP-адрес Raspberry Pi:**

```bash
hostname -I
```

Допустим, получен адрес `172.20.10.2`.

**2. Откройте браузер** на телефоне или ноутбуке и перейдите по адресу:

```
http://172.20.10.2:5000
```

В интерфейсе будет отображаться поток с камеры вашего автомобиля. Движок **OpenCV DNN** будет обрабатывать дорожную обстановку в реальном времени — отслеживайте стабильный FPS в левом углу экрана!

---

## Команды оперативного управления

| Действие | Команда |
|---|---|
| Просмотр логов (детекции, FPS) | `docker logs -f road_sign_app` |
| Остановить контейнер | `docker stop road_sign_app` |
| Запустить контейнер снова | `docker start road_sign_app` |
| Пересобрать и перезапустить | `docker stop road_sign_app && docker rm road_sign_app`, затем Шаг 1 и Шаг 2 |

---

## Быстрая проверка работоспособности

```bash
# Статус контейнера
docker ps -a --filter name=road_sign_app

# Последние 50 строк логов
docker logs --tail=50 road_sign_app

# Убедиться, что порт 5000 слушается
ss -tlnp | grep 5000
```

