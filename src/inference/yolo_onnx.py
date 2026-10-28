from typing import List
import numpy as np
from ultralytics import YOLO
from src.inference.base import BaseDetector, Detection


class YOLODetector(BaseDetector):
    """
    Детектор людей на базе YOLOv8n / YOLOv10n (Ultralytics PyTorch / ONNX Runtime)
    для этапа разработки и валидации MVP (Stage I).
    Поддерживает как стандартные веса (yolov8n.pt / yolov8n.onnx),
    так и дообученные веса (например, под камеру строго сверху).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        person_class_id: int = 0,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        self.model = YOLO(model_path)

    def infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)

        for xyxy, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
            x1, y1, x2, y2 = map(float, xyxy)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    conf=float(conf),
                    class_id=int(cls_id),
                )
            )
        return detections

    def export_to_onnx(self, imgsz: int = 640) -> str:
        """
        Экспорт модели в ONNX для последующей компиляции через Hailo DFC в .hef.
        """
        return str(self.model.export(format="onnx", imgsz=imgsz, simplify=True))
