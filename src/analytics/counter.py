import time
from typing import Any, Dict, List, Sequence, Tuple
import cv2
import numpy as np


def _ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Векторное произведение (ориентированная площадь) для проверки пересечения отрезков."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    d1 = _ccw(p1, p2, q1)
    d2 = _ccw(p1, p2, q2)
    d3 = _ccw(q1, q2, p1)
    d4 = _ccw(q1, q2, p2)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    )


class TripwireAndQueueAnalytics:
    """
    Модуль подсчета трафика через виртуальные линии (Tripwire Counter)
    и плотности очередей у касс (Queue Density).
    """

    def __init__(
        self,
        tripwires_config: Sequence[Dict[str, Any]],
        queue_zones_config: Sequence[Dict[str, Any]],
        default_queue_threshold: int = 3,
    ):
        self.tripwires: List[Dict[str, Any]] = []
        for tw in tripwires_config:
            self.tripwires.append(
                {
                    "line_id": tw["line_id"],
                    "name": tw.get("name", tw["line_id"]),
                    "start": (float(tw["start"][0]), float(tw["start"][1])),
                    "end": (float(tw["end"][0]), float(tw["end"][1])),
                    "in_count": 0,
                    "out_count": 0,
                }
            )

        self.queue_zones: List[Dict[str, Any]] = []
        for qz in queue_zones_config:
            self.queue_zones.append(
                {
                    "zone_id": qz["zone_id"],
                    "name": qz.get("name", qz["zone_id"]),
                    "max_people_threshold": int(
                        qz.get("max_people_threshold", default_queue_threshold)
                    ),
                    "polygon": np.array(qz["polygon"], dtype=np.float32),
                }
            )

        # История предыдущих позиций треков {track_id: (x_m, y_m)}
        self._prev_positions: Dict[int, Tuple[float, float]] = {}
        # Защита от дребезга при пересечении линии {(track_id, line_id): timestamp}
        self._last_crossing_time: Dict[Tuple[int, str], float] = {}

    def update(
        self,
        active_tracks: Sequence[Tuple[int, Tuple[float, float]]],
        current_timestamp: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """
        Обновляет пересечения линий и очереди в зонах касс.
        Возвращает:
          1) список событий (tripwire_crossed, queue_overflow)
          2) статистику по линиям
          3) статистику по очередям
        """
        if current_timestamp <= 0:
            current_timestamp = time.time()

        events: List[Dict[str, Any]] = []
        active_ids = {tid for tid, _ in active_tracks}

        # 1. Tripwire Counter
        for tid, curr_pos in active_tracks:
            if tid in self._prev_positions:
                prev_pos = self._prev_positions[tid]
                for tw in self.tripwires:
                    l_start = tw["start"]
                    l_end = tw["end"]
                    if _segments_intersect(prev_pos, curr_pos, l_start, l_end):
                        cross_key = (int(tid), str(tw["line_id"]))
                        last_t = self._last_crossing_time.get(cross_key, 0.0)
                        if (current_timestamp - last_t) < 0.8:
                            continue
                        self._last_crossing_time[cross_key] = current_timestamp

                        side = _ccw(l_start, l_end, curr_pos)
                        direction = "in" if side >= 0 else "out"
                        if direction == "in":
                            tw["in_count"] += 1
                        else:
                            tw["out_count"] += 1

                        events.append(
                            {
                                "event_type": "tripwire_crossed",
                                "line_id": tw["line_id"],
                                "track_id": int(tid),
                                "direction": direction,
                                "in_count": tw["in_count"],
                                "out_count": tw["out_count"],
                                "timestamp": int(current_timestamp),
                            }
                        )
            self._prev_positions[tid] = curr_pos

        # Очистка старых треков
        stale = [tid for tid in self._prev_positions if tid not in active_ids]
        for tid in stale:
            del self._prev_positions[tid]
            for key in [k for k in self._last_crossing_time if k[0] == tid]:
                del self._last_crossing_time[key]

        tripwire_stats = {
            tw["line_id"]: {
                "line_id": tw["line_id"],
                "name": tw["name"],
                "in_count": tw["in_count"],
                "out_count": tw["out_count"],
                "total_crossings": tw["in_count"] + tw["out_count"],
            }
            for tw in self.tripwires
        }

        # 2. Queue Density
        queue_stats: Dict[str, Any] = {}
        for qz in self.queue_zones:
            z_id = qz["zone_id"]
            poly = qz["polygon"]
            threshold = qz["max_people_threshold"]
            in_queue: List[int] = []

            for tid, (x_m, y_m) in active_tracks:
                if cv2.pointPolygonTest(poly, (float(x_m), float(y_m)), False) >= 0:
                    in_queue.append(int(tid))

            is_overflow = len(in_queue) >= threshold
            if is_overflow:
                events.append(
                    {
                        "event_type": "queue_alert",
                        "zone_id": z_id,
                        "people_count": len(in_queue),
                        "threshold": threshold,
                        "track_ids": in_queue,
                        "timestamp": int(current_timestamp),
                    }
                )

            queue_stats[z_id] = {
                "zone_id": z_id,
                "name": qz["name"],
                "people_count": len(in_queue),
                "threshold": threshold,
                "is_overflow": is_overflow,
                "track_ids": in_queue,
            }

        return events, tripwire_stats, queue_stats
