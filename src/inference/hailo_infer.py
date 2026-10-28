from typing import List
import cv2
import numpy as np
from src.inference.base import BaseDetector, Detection


class HailoDetector(BaseDetector):
    """
    Детектор для целевого оборудования Этапа II:
    Ускоритель ASUS UGen300 (Hailo-10H, 40 TOPS INT4, 8 GB VRAM).
    Использует HailoRT (hailo_platform) и скомпилированный бинарник .hef.

    Пайплайн компиляции модели:
    Model (ONNX) -> Hailo DFC (Parsing & Calibration) -> Model Allocation -> .hef
    """

    def __init__(
        self,
        hef_path: str = "models/yolov8n_hailo10h.hef",
        conf_threshold: float = 0.35,
        person_class_id: int = 0,
        input_size: int = 640,
    ):
        self.hef_path = hef_path
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id
        self.input_size = input_size

        try:
            import hailo_platform as hpf  # type: ignore
            self._hpf = hpf
            self.hef = hpf.HEF(self.hef_path)
            self.target = hpf.VDevice()
            configure_params = hpf.ConfigureParams.create_from_hef(
                hef=self.hef, interface=hpf.HailoStreamInterface.PCIe
            )
            self.network_groups = self.target.configure(self.hef, configure_params)
            self.network_group = self.network_groups[0]
            self.input_vstreams_params = hpf.InputVStreamParams.make(
                self.network_group, format_type=hpf.FormatType.UINT8
            )
            self.output_vstreams_params = hpf.OutputVStreamParams.make(
                self.network_group, format_type=hpf.FormatType.FLOAT32
            )
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        except ImportError as exc:
            raise RuntimeError(
                "Пакет 'hailo_platform' (HailoRT) не найден. "
                "Для этапа разработки (Stage I) используйте backend: 'yolo' в configs/config.yaml. "
                "Для запуска на ASUS UGen300 (Hailo-10H) установите HailoRT SDK."
            ) from exc

    def infer(self, frame: np.ndarray) -> List[Detection]:
        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = {self.input_vstream_info.name: np.expand_dims(rgb, axis=0)}

        with self.network_group.activate(self.network_group.create_params()):
            with self._hpf.InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as infer_pipeline:
                raw_outputs = infer_pipeline.infer(input_data)

        detections: List[Detection] = []
        # Пост-обработка выхода Hailo NMS (формат [ymin, xmin, ymax, xmax, score] по классам)
        for _, output_tensor in raw_outputs.items():
            person_boxes = output_tensor[0][self.person_class_id]
            for det in person_boxes:
                if len(det) < 5:
                    continue
                ymin, xmin, ymax, xmax, score = det[:5]
                if score < self.conf_threshold:
                    continue
                x1 = float(xmin * w_orig)
                y1 = float(ymin * h_orig)
                x2 = float(xmax * w_orig)
                y2 = float(ymax * h_orig)
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        conf=float(score),
                        class_id=self.person_class_id,
                    )
                )
        return detections
