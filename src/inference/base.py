from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """
    Структура единичной детекции человека на кадре.
    bbox: (x1, y1, x2, y2) в пикселях
    conf: уверенность модели [0.0 .. 1.0]
    class_id: идентификатор класса (0 = person в COCO)
    track_id: присваивается модулем трекинга (ByteTrack)
    """
    bbox: Tuple[float, float, float, float]
    conf: float
    class_id: int = 0
    track_id: Optional[int] = None

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Нижняя центральная точка bounding box (координаты стоп [u, v] при съемке под углом)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))

    @property
    def center(self) -> Tuple[float, float]:
        """Геометрический центр bounding box (актуально для камер строго вертикального вида сверху)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def get_anchor_point(self, mode: str = "bottom_center") -> Tuple[float, float]:
        if mode == "center":
            return self.center
        return self.bottom_center


class BaseDetector(ABC):
    """
    Абстрактный интерфейс детектора для прозрачного переключения
    между этапом I (YOLOv8 PyTorch/ONNX) и этапом II (ASUS UGen300 Hailo-10H .hef).
    """

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Возвращает список детекций: bbox, conf, class_id."""
        pass
