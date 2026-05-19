import cv2
import time
from flask import Flask, render_template, Response
from config import InferenceConfig
from detector import TrafficSignDetector

app = Flask(__name__)

# Инициализируем конфигурацию и модуль детектора
config = InferenceConfig()
detector = TrafficSignDetector(config=config)

def generate_frames():
    # Инициализация камеры (0 — дефолтный индекс для Малины)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Прогоняем кадр через детектор
        frame = detector.detect_and_draw(frame)
        
        # Расчет системного FPS видеопотока
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Кодируем кадр в JPEG для передачи на веб-страницу
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Запускаем сервер на порту 5000, открытом для локальной сети автомобиля
    app.run(host='0.0.0.0', port=5000, debug=False)