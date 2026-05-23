import cv2
import time
import json
import numpy as np
from pathlib import Path
from flask import Flask, render_template, Response
from config import InferenceConfig
from detector import TrafficSignDetector

app = Flask(__name__)

# Инициализируем конфигурацию и оптимизированный модуль детектора
config = InferenceConfig()
detector = TrafficSignDetector(config=config)

def generate_frames():
    print("[INFO] Инициализация USB-камеры...")
    
    # Открываем камеру через прямой путь Linux-устройства в обход багов индексации Докера
    cap = cv2.官方_V4L2 = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    
    # Критически важно для промышленных ELP-камер: включаем аппаратное сжатие MJPEG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Даем матрице 1 секунду на физический прогрев экспозиции и фокуса
    time.sleep(1.0)
    
    if not cap.isOpened():
        print("[ERROR] Критическая ошибка: Камера /dev/video0 аппаратно недоступна!")
        return
        
    prev_time = 0
    print("[INFO] Трансляция видеопотока автомобиля успешно начата!")
    
    while True:
        ret, frame = cap.read()
        
        # Если кадр битый или пустой — пропускаем его и ждем следующий, не ломая стрим!
        if not ret or frame is None:
            print("[WARN] Камера пропустила кадр, ожидание следующего...")
            time.sleep(0.03)  # небольшая пауза, соответствующая ~30 FPS
            continue
            
        try:
            # Запускаем обработку кадра нашим C++ ядром OpenCV DNN
            frame = detector.detect_and_draw(frame)
            
            # Расчет и отрисовка реального системного FPS видеопотока
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Бинарное кодирование кадра в MJPEG-поток для браузера
            ret_encode, buffer = cv2.imencode('.jpg', frame)
            if not ret_encode:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
        except Exception as e:
            print(f"[ERROR] Ошибка во время обработки кадра: {e}")
            continue

    cap.release()

@app.route('/')
def index():
    # Отдаем HTML-страницу
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Роут для бесконечного MJPEG-стриминга видео
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Запускаем сервер на порту 5000, открытом для локальной сети автомобиля
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)