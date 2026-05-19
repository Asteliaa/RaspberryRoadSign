import os
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] 

MODEL_IN = PROJECT_ROOT / "models" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "app" / "weights"

def main():
    
    if not MODEL_IN.exists():
        print("Файл не найден")
        sys.exit(1)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(str(MODEL_IN))
    print(f"Модель: {model.task}")
    
    exported_path_str = model.export(
        format="onnx",
        imgsz=320,        
        half=False,       
        simplify=True,    
        dynamic=False,    # Фиксированный размер кадра 
        opset=12    
    )
    
    exported_file = Path(exported_path_str)
    target_file = OUTPUT_DIR / "best.onnx"

    if target_file.exists():
        os.remove(target_file)
        
    shutil.move(str(exported_file), str(target_file))
    
    print(f"Экспорт завершен")

if __name__ == "__main__":
    main()