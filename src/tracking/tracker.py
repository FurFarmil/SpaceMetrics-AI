import warnings
from typing import List
import numpy as np
import supervision as sv
from src.inference.base import Detection


class ObjectTracker:
    """
    Обертка над алгоритмом ByteTrack (через библиотеку supervision).
    Присваивает устойчивый уникальный track_id каждому обнаруженному человеку.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate,
            )

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Принимает список детекций текущего кадра и возвращает
        отфильтрованный список детекций с присвоенными track_id.
        """
        if not detections:
            self.byte_tracker.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.conf for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked_dets = self.byte_tracker.update_with_detections(sv_dets)

        result: List[Detection] = []
        if tracked_dets.tracker_id is None or len(tracked_dets) == 0:
            return result

        for bbox, conf, cls_id, trk_id in zip(
            tracked_dets.xyxy,
            tracked_dets.confidence,
            tracked_dets.class_id,
            tracked_dets.tracker_id,
        ):
            if trk_id is None:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            result.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    conf=float(conf),
                    class_id=int(cls_id),
                    track_id=int(trk_id),
                )
            )
        return result

    def reset(self) -> None:
        self.byte_tracker.reset()
