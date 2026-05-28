import cv2
import time
import platform
from flask import Flask, render_template, Response, jsonify
from config import InferenceConfig
from detector import TrafficSignDetector

app = Flask(__name__)

config = InferenceConfig()
detector = TrafficSignDetector(config=config)

current_mode = "both"


def get_camera():
    current_os = platform.system()
    if current_os == "Linux":
        cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def generate_frames():
    global current_mode
    cap = get_camera()
    time.sleep(1.0)

    if not cap.isOpened():
        print("[ERROR] Камера недоступна!")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        try:
            # Детектор отрабатывает по выбранному режиму (без Ultralytics!)
            frame = detector.detect_and_draw(frame, mode=current_mode)

            # Считаем FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Рисуем FPS в углу
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Кодируем и отправляем
            ret_encode, buffer = cv2.imencode('.jpg', frame)
            if not ret_encode:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"[ERROR] Ошибка в цикле видеопотока: {e}")
            continue

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    global current_mode
    if mode in ["model1", "model2", "both"]:
        current_mode = mode
        return jsonify({"status": "success", "mode": current_mode}), 200
    return jsonify({"status": "error"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
