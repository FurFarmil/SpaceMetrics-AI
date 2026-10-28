from collections import deque
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from src.geometry.homography import HomographyProjector
from src.inference.base import Detection


# Палитра контрастных неоновых цветов для устойчивой визуализации треков (BGR)
TRACK_COLORS = [
    (255, 229, 0),    # Cyan / Голубой
    (129, 185, 16),   # Emerald / Изумрудный
    (11, 158, 245),   # Amber / Янтарный
    (246, 92, 139),   # Purple / Фиолетовый
    (153, 72, 236),   # Magenta / Розовый
    (246, 130, 59),   # Blue / Синий
    (166, 184, 20),   # Teal / Бирюзовый
    (22, 115, 249),   # Orange / Оранжевый
]


def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


class StreamVisualizer:
    """
    Модуль генерации аннотированного видеопотока Live View в реальном времени.
    Отрисовывает:
      - Bounding box с угловыми засечками и бейджем #ID + Conf
      - Точку контакта стоп с полом (Foot Anchor) и физические координаты (X, Y) в метрах
      - Плавные шлейфы траекторий движения человека (Motion Trails)
      - Наложенные зоны витрин (Dwell) и линии входа (Tripwires) со стрелками направления
      - Информационный Heads-Up Display (HUD) с метриками и статусом Zero-Video
    """

    def __init__(self, max_trail_len: int = 24):
        self.max_trail_len = max_trail_len
        self.trails: Dict[int, deque] = {}  # {track_id: deque([(u, v), ...])}
        self.track_last_seen: Dict[int, int] = {}

    def update_trails(self, tracked_objects: Sequence[Detection], frame_idx: int) -> None:
        """Обновляет историю точек стоп для отрисовки траекторий."""
        active_ids = set()
        for det in tracked_objects:
            if det.track_id is None:
                continue
            tid = int(det.track_id)
            active_ids.add(tid)
            u, v = det.get_anchor_point(mode="bottom_center")
            pt = (int(round(u)), int(round(v)))

            if tid not in self.trails:
                self.trails[tid] = deque(maxlen=self.max_trail_len)
            self.trails[tid].append(pt)
            self.track_last_seen[tid] = frame_idx

        # Очистка неактивных треков (не появлявшихся более 40 кадров)
        stale = [
            tid for tid, last_f in self.track_last_seen.items()
            if tid not in active_ids and (frame_idx - last_f) > 40
        ]
        for tid in stale:
            self.trails.pop(tid, None)
            self.track_last_seen.pop(tid, None)

    def annotate(
        self,
        frame: np.ndarray,
        tracked_objects: Sequence[Detection],
        frame_idx: int = 0,
        fps: float = 0.0,
        projector: Optional[HomographyProjector] = None,
        zones_config: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        overlay_options: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        """Аннотирует кадр и возвращает результирующее BGR-изображение."""
        options = overlay_options or {
            "boxes": True,
            "trails": True,
            "anchors": True,
            "zones": True,
            "hud": True,
        }

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        self.update_trails(tracked_objects, frame_idx)

        # 1. Отрисовка зон витрин и линий входа прямо на видеокадре
        if options.get("zones", True) and zones_config and projector:
            self._draw_zones_and_tripwires(annotated, zones_config, projector)

        # 2. Отрисовка шлейфов траекторий движения (Motion Trails)
        if options.get("trails", True):
            for det in tracked_objects:
                if det.track_id is None:
                    continue
                tid = int(det.track_id)
                pts = list(self.trails.get(tid, []))
                if len(pts) < 2:
                    continue
                color = _get_track_color(tid)
                for i in range(len(pts) - 1):
                    # Прогрессивная толщина от хвоста к текущей точке
                    thickness = max(1, int(1 + (i / len(pts)) * 2.5))
                    cv2.line(annotated, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

        # 3. Отрисовка Bounding Boxes, ID и точек контакта с полом
        for det in tracked_objects:
            if det.track_id is None:
                continue
            tid = int(det.track_id)
            color = _get_track_color(tid)
            x1, y1, x2, y2 = map(int, det.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            u, v = det.get_anchor_point(mode="bottom_center")
            u_px, v_px = int(round(u)), int(round(v))

            # Координаты в метрах пола
            xm, ym = 0.0, 0.0
            if projector:
                xm, ym = projector.pixel_to_meter(u, v)

            # 3.1 Bounding box с угловыми засечками
            if options.get("boxes", True):
                # Тонкая рамка
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

                # Угловые засечки (Corner brackets)
                bw, bh = x2 - x1, y2 - y1
                clen = max(6, min(18, bw // 4, bh // 4))
                cthick = 3
                # Верхний левый угол
                cv2.line(annotated, (x1, y1), (x1 + clen, y1), color, cthick, cv2.LINE_AA)
                cv2.line(annotated, (x1, y1), (x1, y1 + clen), color, cthick, cv2.LINE_AA)
                # Верхний правый угол
                cv2.line(annotated, (x2, y1), (x2 - clen, y1), color, cthick, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2, y1 + clen), color, cthick, cv2.LINE_AA)
                # Нижний левый угол
                cv2.line(annotated, (x1, y2), (x1 + clen, y2), color, cthick, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1, y2 - clen), color, cthick, cv2.LINE_AA)
                # Нижний правый угол
                cv2.line(annotated, (x2, y2), (x2 - clen, y2), color, cthick, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2, y2 - clen), color, cthick, cv2.LINE_AA)

                # Бейдж с ID и уверенностью
                label = f"#{tid} ({int(det.conf * 100)}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                by1 = max(0, y1 - th - 8)
                by2 = y1
                bx2 = min(w - 1, x1 + tw + 10)
                cv2.rectangle(annotated, (x1, by1), (bx2, by2), color, -1)
                # Контрастный текст
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, by2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (10, 15, 25),
                    1,
                    cv2.LINE_AA,
                )

            # 3.2 Точка контакта стоп с полом (Foot Anchor)
            if options.get("anchors", True):
                # Внешний круг
                cv2.circle(annotated, (u_px, v_px), 6, color, 2, cv2.LINE_AA)
                # Внутренняя яркая точка
                cv2.circle(annotated, (u_px, v_px), 2, (255, 255, 255), -1, cv2.LINE_AA)

                # Подпись физических координат в метрах
                coord_txt = f"{xm:.1f}m, {ym:.1f}m"
                (cw, ch), _ = cv2.getTextSize(coord_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cx = max(4, min(w - cw - 4, u_px - cw // 2))
                cy = min(h - 6, v_px + ch + 10)
                # Тень текста для читаемости на любом фоне
                cv2.putText(annotated, coord_txt, (cx + 1, cy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated, coord_txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Верхний информационный HUD (Heads-Up Display)
        if options.get("hud", True):
            self._draw_hud(annotated, frame_idx, fps, tracked_objects, telemetry)

        return annotated

    def _draw_zones_and_tripwires(
        self,
        frame: np.ndarray,
        zones_config: Dict[str, Any],
        projector: HomographyProjector,
    ) -> None:
        """Отрисовывает калибровочные границы, витрины, кассы и линии входа с точностью до пикселя."""
        # 0. Калибровочный периметр пола (если задан)
        h_calib = zones_config.get("homography_calibration", {})
        if "src_pixel_points" in h_calib and len(h_calib["src_pixel_points"]) == 4:
            pts_bound = np.array(h_calib["src_pixel_points"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_bound], isClosed=True, color=(75, 85, 99), thickness=1, lineType=cv2.LINE_AA)

        # 1. Витрины (Dwell zones) - #00e5ff -> BGR (255, 229, 0)
        for dz in zones_config.get("dwell_zones", []):
            if dz.get("pixel_polygon") and len(dz["pixel_polygon"]) >= 3:
                pix_pts = [(int(p[0]), int(p[1])) for p in dz["pixel_polygon"]]
            else:
                pts_m = dz.get("polygon", [])
                if len(pts_m) < 3:
                    continue
                pix_pts = [projector.meter_to_pixel(p[0], p[1]) for p in pts_m]

            if len(pix_pts) < 3:
                continue

            pts_arr = np.array(pix_pts, dtype=np.int32).reshape((-1, 1, 2))

            # Полупрозрачная заливка зоны
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_arr], (255, 229, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Контур зоны
            cv2.polylines(frame, [pts_arr], isClosed=True, color=(255, 229, 0), thickness=2, lineType=cv2.LINE_AA)

            # Подпись зоны
            p0 = pix_pts[0]
            name = dz.get("name", dz.get("zone_id", "Витрина"))
            t_sec = dz.get("threshold_seconds", 1.5)
            cv2.putText(
                frame,
                f"* {name} ({t_sec}s)",
                (p0[0] + 6, max(18, p0[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 229, 0),
                1,
                cv2.LINE_AA,
            )

        # 2. Зоны очередей касс (Queue zones) - #ef4444 -> BGR (68, 68, 239)
        for qz in zones_config.get("queue_zones", []):
            if qz.get("pixel_polygon") and len(qz["pixel_polygon"]) >= 3:
                pix_pts = [(int(p[0]), int(p[1])) for p in qz["pixel_polygon"]]
            else:
                pts_m = qz.get("polygon", [])
                if len(pts_m) < 3:
                    continue
                pix_pts = [projector.meter_to_pixel(p[0], p[1]) for p in pts_m]

            if len(pix_pts) < 3:
                continue

            pts_arr = np.array(pix_pts, dtype=np.int32).reshape((-1, 1, 2))

            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_arr], (68, 68, 239))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            cv2.polylines(frame, [pts_arr], isClosed=True, color=(68, 68, 239), thickness=2, lineType=cv2.LINE_AA)

            p0 = pix_pts[0]
            name = qz.get("name", qz.get("zone_id", "Касса"))
            cv2.putText(
                frame,
                f"[Kassa] {name}",
                (p0[0] + 6, max(18, p0[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (68, 68, 239),
                1,
                cv2.LINE_AA,
            )

        # 3. Линии входа (Tripwires) - #10b981 -> BGR (129, 185, 16)
        for tw in zones_config.get("tripwires", []):
            if tw.get("pixel_start") and tw.get("pixel_end"):
                p1 = (int(tw["pixel_start"][0]), int(tw["pixel_start"][1]))
                p2 = (int(tw["pixel_end"][0]), int(tw["pixel_end"][1]))
            else:
                p1_m = tw.get("start", [1.0, 5.0])
                p2_m = tw.get("end", [11.0, 5.0])
                p1 = projector.meter_to_pixel(p1_m[0], p1_m[1])
                p2 = projector.meter_to_pixel(p2_m[0], p2_m[1])

            # Основная линия входа
            cv2.line(frame, p1, p2, (129, 185, 16), 3, cv2.LINE_AA)

            # Окружности начала (1) и конца (2)
            cv2.circle(frame, p1, 5, (129, 185, 16), -1, cv2.LINE_AA)
            cv2.circle(frame, p2, 5, (129, 185, 16), -1, cv2.LINE_AA)
            cv2.putText(frame, "1", (p1[0] - 3, max(14, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "2", (p2[0] - 3, max(14, p2[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # Стрелка стороны входа (перпендикуляр)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length > 10:
                nx, ny = -dy / length, dx / length
                ax = int(mx + nx * 22)
                ay = int(my + ny * 22)
                cv2.arrowedLine(frame, (mx, my), (ax, ay), (153, 211, 52), 2, cv2.LINE_AA, tipLength=0.35)
                cv2.putText(frame, "IN", (ax + 4, ay + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (153, 211, 52), 1, cv2.LINE_AA)

            name = tw.get("name", tw.get("line_id", "Линия"))
            cv2.putText(frame, f"- {name}", (mx + 8, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (129, 185, 16), 1, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        tracked_objects: Sequence[Detection],
        telemetry: Optional[Dict[str, Any]],
    ) -> None:
        """Отрисовывает верхнюю плашку с телеметрией в стиле кибер-сенсора."""
        h, w = frame.shape[:2]
        hud_h = 36

        # Полупрозрачная черная подложка
        sub = frame[0:hud_h, 0:w].copy()
        cv2.rectangle(sub, (0, 0), (w, hud_h), (9, 13, 22), -1)
        cv2.addWeighted(sub, 0.85, frame[0:hud_h, 0:w], 0.15, 0, frame[0:hud_h, 0:w])
        cv2.line(frame, (0, hud_h), (w, hud_h), (30, 41, 59), 1)

        # Зеленый индикатор трансляции
        cv2.circle(frame, (18, hud_h // 2), 5, (16, 185, 129), -1, cv2.LINE_AA)
        cv2.putText(frame, "LIVE TRACKING", (30, hud_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)

        # Метрики FPS, людей и посетителей
        active_cnt = len([d for d in tracked_objects if d.track_id is not None])
        fps_txt = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        tracks_txt = f"PEOPLE: {active_cnt}"

        total_in, total_out = 0, 0
        if telemetry and "tripwires" in telemetry:
            for tw in telemetry["tripwires"].values():
                total_in += tw.get("in_count", 0)
                total_out += tw.get("out_count", 0)
        cross_txt = f"IN: {total_in} | OUT: {total_out}"

        cv2.putText(frame, fps_txt, (180, hud_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, tracks_txt, (280, hud_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, cross_txt, (400, hud_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 211, 153), 1, cv2.LINE_AA)

        badge_txt = "ZERO-VIDEO SENSOR | ASUS UGen300"
        (bw, _), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, badge_txt, (w - bw - 14, hud_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (148, 163, 184), 1, cv2.LINE_AA)
