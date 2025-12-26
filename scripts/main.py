import cv2
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO


try:
    from rtsd_to_by import ID_TO_BY
except ImportError:
    print("Ошибка: Не найден")
    sys.exit(1)

MODEL_PATH = "/home/ruslana/Projects/RaspberryYolo/RaspberryRoadSign/runs/rtsd_train/rtsd_yolo11n_pi_other50/weights/best.pt"
VIDEO_PATH = "/home/ruslana/Projects/RaspberryYolo/RaspberryRoadSign/scripts/video_test/driving_lesson_11-13min.mp4"
OUTPUT_PATH = "/home/ruslana/Projects/RaspberryYolo/RaspberryRoadSign/scripts/video_predict/result_11-13min_2.mp4"

CONF_THRESHOLD = 0.35  # Порог уверенности

def draw_transparent_box(img, x1, y1, x2, y2, color, alpha=0.4):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


if not Path(MODEL_PATH).exists():
    print(f"Модель не найдена: {MODEL_PATH}")
    sys.exit(1)

print("Загрузка модели:")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Ошибка открытия видео: {VIDEO_PATH}")
    sys.exit(1)

# Исходные параметры видео
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Видео: {width}x{height} @ {fps} FPS")
print(f"Inference Size будет установлен автоматически под размер видео")

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # type: ignore

np.random.seed(42)
colors = np.random.uniform(0, 255, size=(155, 3))

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1

        results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=width, verbose=False)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in ID_TO_BY:
                    sign_code = ID_TO_BY[cls_id]
                    label = f"{sign_code} {conf:.0%}"
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = colors[cls_id]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7  
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                    bg_x1 = x1
                    bg_y1 = y1 - text_h - 10
                    bg_x2 = x1 + text_w + 10
                    bg_y2 = y1
                    
                    if bg_y1 < 0:
                        bg_y1 = y1
                        bg_y2 = y1 + text_h + 10
                        text_y = y1 + text_h + 5
                    else:
                        text_y = y1 - 5

                    draw_transparent_box(frame, bg_x1, bg_y1, bg_x2, bg_y2, color, alpha=0.6)
                    
                    cv2.putText(frame, label, (x1 + 5, text_y), 
                               font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out.write(frame)
        
        if frame_count % 10 == 0:
            percent = (frame_count / total_frames) * 100
            print(f"Processing: {frame_count}/{total_frames} ({percent:.1f}%)")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n Сохранено в: {OUTPUT_PATH}")