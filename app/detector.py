import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from config import InferenceConfig

class TrafficSignDetector:
    def __init__(self, config: InferenceConfig, classes_path="classes.json"):
        self.config = config
        
        # Загружаем наш эталонный JSON
        if Path(classes_path).exists():
            with open(classes_path, "r", encoding="utf-8") as f:
                self.classes_map = json.load(f)
        else:
            print(f"⚠️ Файл {classes_path} не найден! Коды ПДД будут заменены на ID классов.")
            self.classes_map = {}
            
        print(f"[INFO] Загрузка ONNX модели из: {self.config.model_path}")
        # Инициализируем легковесный ONNX-инференс
        self.model = YOLO(str(self.config.model_path), task="detect")
        print("✓ Движок детекции успешно запущен в Docker!")

    def detect_and_draw(self, frame):
        """Прогон кадра через нейросеть и отрисовка чистых ПДД-кодов (без кириллицы)."""
        results = self.model(
            frame, 
            conf=self.config.conf_threshold, 
            iou=self.config.iou_threshold, 
            imgsz=self.config.imgsz, 
            device=self.config.device,
            verbose=False
        )[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Извлекаем полную строку (например: "3.24: Ограничение максимальной скорости")
            full_class_text = self.classes_map.get(str(cls_id), f"ID_{cls_id}: Знак")
            
            # Магия: отрезаем русский текст, оставляем только чистый код ПДД!
            pdd_code = full_class_text.split(":")[0].strip()
            
            # Формируем компактную латинскую подпись для OpenCV
            label = f"{pdd_code} ({conf:.2f})"
            
            # Рисуем рамку и текст стандартными сверхбыстрыми методами OpenCV
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        return frame