from pathlib import Path
from pydantic import BaseModel, Field, field_validator

# Автоматически определяем абсолютный путь к папке app/
BASE_DIR = Path(__file__).resolve().parent


class InferenceConfig(BaseModel):
    """Конфигурация инференса для Raspberry Pi и ПК."""
    model_path_1: Path = Field(
        BASE_DIR / "weights" / "signs.onnx",
        description="Traffic Sign Detection ONNX"
    )
    model_path_2: Path = Field(
        BASE_DIR / "weights" / "traffic_lights.onnx",
        description="Traffic Light Detection ONNX"
    )
    conf_threshold: float = Field(0.45, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    imgsz: int = Field(320, gt=0)
    device: str = Field("cpu")

    @field_validator('model_path_1', 'model_path_2')
    @classmethod
    def validate_model_path(cls, v: Path) -> Path:
        if not v.exists():
            print(f"Файл модели не найден по пути: {v}")
        return v
