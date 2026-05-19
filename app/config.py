from pathlib import Path
from pydantic import BaseModel, Field, field_validator

class InferenceConfig(BaseModel):
    """Конфигурация инференса для Raspberry Pi."""
    model_path: Path = Field(
        Path("weights/best.onnx"), 
        description="Путь к весам ONNX"
    )
    conf_threshold: float = Field(
        0.45, 
        ge=0.0, 
        le=1.0, 
        description="Порог уверенности детекции"
    )
    iou_threshold: float = Field(
        0.45, 
        ge=0.0, 
        le=1.0, 
        description="Порог NMS для перекрывающихся рамок"
    )
    imgsz: int = Field(
        320, 
        gt=0, 
        description="Размер кадра для нейросети (320 для Pi 4 / 640 для Pi 5)"
    )
    device: str = Field(
        "cpu", 
        description="Устройство выполнения (всегда cpu на Малине)"
    )

    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: Path) -> Path:
        path = Path(v)
        if not path.exists():
            print(f"⚠️ Предупреждение: Файл весов {v} не найден локально. Будет использован fallback.")
        return path