import time
from typing import Any, Dict, List, Sequence, Tuple
import cv2
import numpy as np


class DwellTimeTracker:
    """
    Модуль мониторинга времени нахождения посетителей (dwell time)
    внутри полигональных зон интереса (витрины, промо-стенды).
    """

    def __init__(self, zones_config: Sequence[Dict[str, Any]], default_threshold_sec: float = 3.0):
        self.default_threshold_sec = default_threshold_sec
        self.zones: List[Dict[str, Any]] = []
        for z in zones_config:
            poly = np.array(z["polygon"], dtype=np.float32)
            self.zones.append(
                {
                    "zone_id": z["zone_id"],
                    "name": z.get("name", z["zone_id"]),
                    "threshold": float(z.get("threshold_seconds", default_threshold_sec)),
                    "polygon": poly,
                }
            )

        # Состояние: {zone_id: {track_id: {"accumulated_sec": float, "alerted": bool}}}
        self._state: Dict[str, Dict[int, Dict[str, Any]]] = {
            z["zone_id"]: {} for z in self.zones
        }
        self.total_dwell_events: int = 0

    def update(
        self,
        active_tracks: Sequence[Tuple[int, Tuple[float, float]]],
        dt_seconds: float = 1.0 / 30.0,
        current_timestamp: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Принимает список (track_id, (x_m, y_m)) на текущем кадре.
        Возвращает:
          1) список новых/обновленных JSON-событий dwell_detected
          2) сводную статистику по каждой зоне
        """
        if current_timestamp <= 0:
            current_timestamp = time.time()

        events: List[Dict[str, Any]] = []
        zone_summary: Dict[str, Dict[str, Any]] = {}
        active_ids = {trk_id for trk_id, _ in active_tracks}

        for zone in self.zones:
            z_id = zone["zone_id"]
            poly = zone["polygon"]
            threshold = zone["threshold"]
            z_tracks = self._state[z_id]

            occupants: List[int] = []
            max_dwell = 0.0

            for trk_id, (x_m, y_m) in active_tracks:
                inside = cv2.pointPolygonTest(poly, (float(x_m), float(y_m)), False) >= 0
                if inside:
                    occupants.append(trk_id)
                    if trk_id not in z_tracks:
                        z_tracks[trk_id] = {"accumulated_sec": 0.0, "alerted": False}
                    z_tracks[trk_id]["accumulated_sec"] += dt_seconds
                    dwell_sec = round(z_tracks[trk_id]["accumulated_sec"], 2)
                    max_dwell = max(max_dwell, dwell_sec)

                    if dwell_sec >= threshold:
                        if not z_tracks[trk_id]["alerted"]:
                            z_tracks[trk_id]["alerted"] = True
                            self.total_dwell_events += 1
                        events.append(
                            {
                                "event_type": "dwell_detected",
                                "track_id": int(trk_id),
                                "zone_id": z_id,
                                "dwell_time_seconds": dwell_sec,
                                "timestamp": int(current_timestamp),
                            }
                        )

            # Очистка ушедших треков
            stale_ids = [tid for tid in z_tracks if tid not in active_ids]
            for tid in stale_ids:
                del z_tracks[tid]

            zone_summary[z_id] = {
                "zone_id": z_id,
                "name": zone["name"],
                "active_tracks": occupants,
                "count": len(occupants),
                "max_dwell_seconds": max_dwell,
            }

        return events, zone_summary
