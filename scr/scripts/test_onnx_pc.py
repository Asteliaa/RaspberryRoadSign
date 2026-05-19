# src/scripts/test_onnx_pc.py (Тест инференса ONNX на ПК)
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# Настройки путей относительно корня
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Тестируем именно тот ONNX, который сгенерировал верхний скрипт
ONNX_MODEL_PATH = PROJECT_ROOT / "app" / "weights" / "best.onnx"
VIDEO_PATH = PROJECT_ROOT / "videos" / "EVE01389.AVI"  # Укажи свое тестовое видео

def main():
    print(f"🔍 Тестируем чтение ONNX-модели...")
    if not ONNX_MODEL_PATH.exists():
        print(f"❌ Файл {ONNX_MODEL_PATH} не найден! Сначала запусти экспорт.")
        return
        
    # Загружаем модель. Ultralytics автоматически поймет, что это ONNX 
    # и запустит его через легковесный движок onnxruntime
    print("[INFO] Загрузка ONNX весов в ядро инференса...")
    model = YOLO(str(ONNX_MODEL_PATH), task="detect")
    print("✓ Модель успешно инициализирована!")
    
    if not VIDEO_PATH.exists():
        print(f"⚠️ Видео {VIDEO_PATH} не найдено, запускаем тест на веб-камере (ID 0)...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"🎬 Открываем тестовый видеофайл: {VIDEO_PATH.name}")
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        
    prev_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🏁 Видеофайл завершился или камера отключена.")
            break
            
        # Прогоняем кадр через ONNX
        t_start = time.time()
        results = model(frame, conf=0.45, verbose=False)[0]
        t_end = time.time()
        
        # Отрисовка рамок знаков
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # На ПК берем дефолтное имя из весов (для быстрой проверки)
            cls_name = results.names[cls_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {cls_id}: {cls_name} {conf:.2f}", (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Расчет FPS
        fps = 1 / (t_end - t_start)
        cv2.putText(frame, f"ONNX FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("PC ONNX Test Window (Press 'Q' to Exit)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Тест инференса успешно завершен.")

if __name__ == "__main__":
    main()