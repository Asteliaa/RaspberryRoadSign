import cv2
import json
import numpy as np
from pathlib import Path
from config import InferenceConfig

class TrafficSignDetector:
    def __init__(self, config: InferenceConfig):
        self.config = config

        # Точный путь к папке app/
        base_dir = Path(__file__).resolve().parent

        # Сверяем пути строго по твоему скриншоту: папка classes/ и точные имена файлов
        self.classes_map_1 = self._load_json(base_dir / "classes" / "classes_signs.json")
        self.classes_map_2 = self._load_json(base_dir / "classes" / "classes_traffic_lights.json")

        print(f"[INFO] Путь к модели знаков: {self.config.model_path_1}")
        self.net1 = cv2.dnn.readNetFromONNX(str(self.config.model_path_1))

        print(f"[INFO] Путь к модели светофоров: {self.config.model_path_2}")
        self.net2 = cv2.dnn.readNetFromONNX(str(self.config.model_path_2))

        for net in [self.net1, self.net2]:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print("✓ Две независимые модели успешно загружены!")

    def _load_json(self, path):
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"⚠️ Ошибка: Файл классов {path} не найден!")
        return {}

    def _process_net(self, net, classes_map, frame, w_img, h_img, color):
        """Прогоняет кадр через конкретную сеть, делает ЛОКАЛЬНЫЙ NMS и возвращает результат с цветом"""
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (self.config.imgsz, self.config.imgsz), swapRB=True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward()
        outputs = np.squeeze(outputs).T

        boxes = []
        confidences = []
        labels = []

        for row in outputs:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            score = classes_scores[class_id]

            if score >= self.config.conf_threshold:
                xc, yc, w, h = row[:4]
                x1 = int((xc - w/2) * w_img / self.config.imgsz)
                y1 = int((yc - h/2) * h_img / self.config.imgsz)
                width = int(w * w_img / self.config.imgsz)
                height = int(h * h_img / self.config.imgsz)

                boxes.append([x1, y1, width, height])
                confidences.append(float(score))
                
                # Извлекаем и чистим имя класса
                full_text = classes_map.get(str(class_id), f"ID_{class_id}")
                pdd_code = full_text.split(":")[0].strip()
                labels.append(f"{pdd_code} ({score:.2f})")

        # Изолированный NMS для текущей модели
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.config.conf_threshold, self.config.iou_threshold)
        
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append({
                    "box": boxes[i],
                    "label": labels[i],
                    "color": color
                })
        return final_detections

    def detect_and_draw(self, frame, mode="both"):
        """Принимает режим работы и наносит рамки нужных цветов на кадр"""
        h_img, w_img = frame.shape[:2]
        detections = []

        # 1. Модель знаков -> ЗЕЛЕНЫЙ ЦВЕТ (0, 255, 0)
        if mode in ["model1", "both"]:
            detections.extend(self._process_net(self.net1, self.classes_map_1, frame, w_img, h_img, color=(0, 255, 0)))

        # 2. Модель светофоров -> ОРАНЖЕВЫЙ ЦВЕТ (0, 145, 255) в формате BGR
        if mode in ["model2", "both"]:
            detections.extend(self._process_net(self.net2, self.classes_map_2, frame, w_img, h_img, color=(0, 145, 255)))

        # Отрисовка всех объектов с их собственными цветами
        for det in detections:
            x, y, w, h = det["box"]
            label = det["label"]
            current_color = det["color"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), current_color, 2)
            cv2.putText(frame, label, (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)

        return frame