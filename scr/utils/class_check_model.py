import json
from ultralytics import YOLO

# Загружаем твою модель на 103 класса
model = YOLO("models/best.pt")

# Извлекаем словарь {id: имя_класса}
model_classes = model.names

print(f"Успешно извлечено классов: {len(model_classes)}")

# Сохраняем в текстовый файл, чтобы мы могли его сразу превратить в mapping.py
with open("docs/model_103_classes.json", "w", encoding="utf-8") as f:
    json.dump(model_classes, f, indent=4, ensure_ascii=False)
    
print("Список сохранен в docs/model_103_classes.json")