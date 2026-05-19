import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

# ---------- настройки ----------
video_path = "videos/EVE01389.AVI"            # входное видео
weights_path = "models/deploy/last.pt"     # твоя модель
output_path = "videos/video_EVE01389.mp4"      # выходное видео с детекцией

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # проверь путь к шрифту
FONT_SIZE = 24
CONF_THR = 0.6

# ---------- функция для русского текста ----------
def draw_russian_text(frame, text, x, y, color=(0, 255, 0)):
    # frame (BGR np.array) -> PIL Image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    # color: BGR -> RGB
    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    # обратно в OpenCV (BGR)
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame

# ---------- загрузка модели ----------
model = YOLO(weights_path)

# ---------- открываем видео ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не могу открыть видео")
    exit()

# параметры исходного видео
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# создаём VideoWriter (кодек и файл)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # можно "XVID" и .avi
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# ---------- основной цикл ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO принимает сразу numpy‑кадр (BGR тоже съедает)
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]

        if conf < CONF_THR:
            continue

        # прямоугольник
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # подпись (русский/любой Unicode через Pillow)
        label = f"{cls_name} {conf:.2f}"
        text_y = max(y1 - 25, 0)
        frame = draw_russian_text(frame, label, x1, text_y)

    # записываем кадр в выходное видео (полный размер)
    out.write(frame)

    # отображаем уменьшенную копию для удобства
    scale = 0.7
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.imshow("YOLO video", small)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------- очистка ----------
cap.release()
out.release()
cv2.destroyAllWindows()
