import asyncio
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse


class MetricsStateStore:
    """
    Потокобезопасное хранилище последних обезличенных метрик, мультимодальных
    тепловых карт и живого кадра камеры для визуальной разметки зон мышкой.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_payload: Dict[str, Any] = {
            "status": "initializing",
            "frame_index": 0,
            "fps": 0.0,
            "active_people_count": 0,
            "floor_tracks": [],
            "dwell_zones": {},
            "tripwires": {},
            "queues": {},
            "recent_events": [],
            "zone_attention": {},
        }
        self._heatmaps: Dict[str, List[List[float]]] = {
            "total": [],
            "dwell": [],
            "flow": [],
        }
        self._zones_config: Dict[str, Any] = {}
        self._rendered_images: Dict[str, bytes] = {}
        self._camera_frame_jpeg: Optional[bytes] = None
        self._annotated_frame_jpeg: Optional[bytes] = None
        self._camera_dims: Tuple[int, int] = (1280, 720)
        self._projector: Any = None
        self._zones_updated = False
        self._overlay_config: Dict[str, bool] = {
            "boxes": True,
            "trails": True,
            "anchors": True,
            "zones": True,
            "hud": True,
        }
        self._init_from_disk()

    def _init_from_disk(self) -> None:
        """Предзагрузка зон и проектора гомографии из файлов конфигурации."""
        try:
            zones_file = "configs/zones.json"
            cfg_file = "configs/config.yaml"
            floor_w, floor_h = 12.0, 10.0
            if os.path.exists(cfg_file):
                import yaml
                with open(cfg_file, "r", encoding="utf-8") as f:
                    y_cfg = yaml.safe_load(f)
                    geo = y_cfg.get("geometry", {})
                    floor_w = float(geo.get("floor_width_meters", 12.0))
                    floor_h = float(geo.get("floor_height_meters", 10.0))
            if os.path.exists(zones_file):
                with open(zones_file, "r", encoding="utf-8") as f:
                    z_cfg = json.load(f)
                    self._zones_config = {
                        "floor_width_meters": floor_w,
                        "floor_height_meters": floor_h,
                        "zones": z_cfg,
                    }
                    h_calib = z_cfg.get("homography_calibration", {})
                    if "src_pixel_points" in h_calib and "dst_meter_points" in h_calib:
                        from src.geometry.homography import HomographyProjector
                        self._projector = HomographyProjector(
                            src_pixel_points=h_calib["src_pixel_points"],
                            dst_meter_points=h_calib["dst_meter_points"],
                        )
        except Exception as e:
            print(f"[MetricsStateStore] Предзагрузка конфигурации: {e}")

    def set_camera_frame(self, jpeg_bytes: bytes, width: int = 1280, height: int = 720) -> None:
        with self._lock:
            self._camera_frame_jpeg = jpeg_bytes
            self._camera_dims = (width, height)

    def get_camera_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._camera_frame_jpeg

    def set_annotated_frame(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._annotated_frame_jpeg = jpeg_bytes

    def get_annotated_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._annotated_frame_jpeg

    def get_overlay_config(self) -> Dict[str, bool]:
        with self._lock:
            return dict(self._overlay_config)

    def set_overlay_config(self, cfg: Dict[str, bool]) -> None:
        with self._lock:
            self._overlay_config.update(cfg)

    def get_camera_dims(self) -> Tuple[int, int]:
        with self._lock:
            return self._camera_dims

    def set_projector(self, projector: Any) -> None:
        with self._lock:
            self._projector = projector

    def get_projector(self) -> Any:
        with self._lock:
            return self._projector

    def signal_zones_updated(self) -> None:
        with self._lock:
            self._zones_updated = True

    def check_and_clear_zones_updated(self) -> bool:
        with self._lock:
            val = self._zones_updated
            self._zones_updated = False
            return val

    def set_zones_config(self, zones_cfg: Dict[str, Any], floor_dims: Dict[str, float]) -> None:
        with self._lock:
            self._zones_config = {
                "floor_width_meters": floor_dims.get("floor_width_meters", 12.0),
                "floor_height_meters": floor_dims.get("floor_height_meters", 10.0),
                "zones": zones_cfg,
            }

    def update(
        self,
        payload: Dict[str, Any],
        heatmaps: Dict[str, List[List[float]]],
        zone_attention: Optional[Dict[str, Any]] = None,
        rendered_images: Optional[Dict[str, bytes]] = None,
    ) -> None:
        with self._lock:
            self._latest_payload = payload
            if zone_attention:
                self._latest_payload["zone_attention"] = zone_attention
            self._heatmaps = heatmaps
            if rendered_images:
                self._rendered_images = rendered_images

    def get_snapshot(self, mode: str = "total") -> Dict[str, Any]:
        with self._lock:
            data = dict(self._latest_payload)
            data["heatmaps"] = self._heatmaps
            data["active_heatmap"] = self._heatmaps.get(mode, self._heatmaps.get("total", []))
            return data

    def get_heatmap_matrix(self, mode: str = "total") -> List[List[float]]:
        with self._lock:
            return self._heatmaps.get(mode, self._heatmaps.get("total", []))

    def get_attention(self) -> Dict[str, Any]:
        with self._lock:
            return self._latest_payload.get("zone_attention", {})

    def get_rendered_image(self, mode: str = "total") -> Optional[bytes]:
        with self._lock:
            return self._rendered_images.get(mode)

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._zones_config)


metrics_store = MetricsStateStore()
app = FastAPI(
    title="SpaceMetrics AI — Zero-Video Edge Telemetry API",
    description="Обезличенные метрики ритейл-пространства и интерактивная разметка зон с камеры",
    version="0.3.0",
)


@app.get("/api/metrics")
async def get_metrics() -> JSONResponse:
    snap = metrics_store.get_snapshot()
    clean_snap = dict(snap)
    clean_snap.pop("heatmaps", None)
    clean_snap.pop("active_heatmap", None)
    return JSONResponse(content=clean_snap)


@app.get("/api/heatmap")
async def get_heatmap(mode: str = Query("total", pattern="^(total|dwell|flow)$")) -> JSONResponse:
    matrix = metrics_store.get_heatmap_matrix(mode=mode)
    return JSONResponse(content={"mode": mode, "heatmap": matrix})


@app.get("/api/heatmap/attention")
async def get_heatmap_attention() -> JSONResponse:
    return JSONResponse(content=metrics_store.get_attention())


@app.get("/api/heatmap/image")
async def get_heatmap_image(mode: str = Query("total", pattern="^(total|dwell|flow)$")) -> Response:
    img_bytes = metrics_store.get_rendered_image(mode=mode)
    if not img_bytes:
        blank = np.full((300, 360, 3), 240, dtype=np.uint8)
        cv2.putText(blank, "Initializing...", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        _, buf = cv2.imencode(".png", blank)
        img_bytes = buf.tobytes()
    return Response(content=img_bytes, media_type="image/png")


@app.get("/api/camera/frame")
async def get_camera_frame() -> Response:
    """Возвращает текущий кадр с камеры для визуального черчения зон мышкой."""
    frame_bytes = metrics_store.get_camera_frame()
    if not frame_bytes:
        w, h = metrics_store.get_camera_dims()
        blank = np.full((h, w, 3), 25, dtype=np.uint8)
        cv2.putText(blank, "Ожидание видеопотока...", (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        _, buf = cv2.imencode(".jpg", blank)
        frame_bytes = buf.tobytes()
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/api/camera/stream")
async def camera_stream() -> StreamingResponse:
    """MJPEG Live View стрим с отрисовкой трекинга людей и зон в реальном времени."""
    async def frame_generator():
        last_frame_bytes = None
        while True:
            frame_bytes = metrics_store.get_annotated_frame()
            if frame_bytes and frame_bytes != last_frame_bytes:
                last_frame_bytes = frame_bytes
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                    + frame_bytes + b"\r\n"
                )
            await asyncio.sleep(0.033)  # ~30 FPS

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/camera/live_frame")
async def get_live_frame() -> Response:
    """Возвращает единичный аннотированный JPEG кадр с трекингом."""
    frame_bytes = metrics_store.get_annotated_frame()
    if not frame_bytes:
        w, h = metrics_store.get_camera_dims()
        blank = np.full((h, w, 3), 25, dtype=np.uint8)
        cv2.putText(blank, "Инициализация Live View...", (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        _, buf = cv2.imencode(".jpg", blank)
        frame_bytes = buf.tobytes()
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/api/camera/overlay_config")
async def get_overlay_config() -> JSONResponse:
    return JSONResponse(content=metrics_store.get_overlay_config())


@app.post("/api/camera/overlay_config")
async def set_overlay_config(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        metrics_store.set_overlay_config(body)
        return JSONResponse(content={"status": "ok", "config": metrics_store.get_overlay_config()})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/config")
async def get_config() -> JSONResponse:
    return JSONResponse(content=metrics_store.get_config())


@app.get("/api/zones/camera_projected")
async def get_zones_camera_projected() -> JSONResponse:
    """
    Возвращает существующие зоны, спроецированные из метров пола в пиксели
    камеры [u, v], чтобы отобразить их прямо поверх видеокадра.
    """
    cfg = metrics_store.get_config()
    zones = cfg.get("zones", {})
    if not zones and os.path.exists("configs/zones.json"):
        with open("configs/zones.json", "r", encoding="utf-8") as f:
            zones = json.load(f)

    projector = metrics_store.get_projector()
    if not projector and "homography_calibration" in zones:
        try:
            from src.geometry.homography import HomographyProjector
            h_calib = zones["homography_calibration"]
            projector = HomographyProjector(
                src_pixel_points=h_calib["src_pixel_points"],
                dst_meter_points=h_calib["dst_meter_points"],
            )
            metrics_store.set_projector(projector)
        except Exception:
            pass

    if not projector:
        return JSONResponse(content={"status": "projector_not_ready", "zones": zones})

    floor_w = float(cfg.get("floor_width_meters", 12.0))
    floor_h = float(cfg.get("floor_height_meters", 10.0))

    # 4 угла пола помещения в пикселях камеры для визуализации границ рабочей зоны
    floor_corners_m = [(0.0, 0.0), (floor_w, 0.0), (floor_w, floor_h), (0.0, floor_h)]
    floor_boundary_pixels = [list(projector.meter_to_pixel(xm, ym)) for xm, ym in floor_corners_m]

    projected_dwell = []
    for dz in zones.get("dwell_zones", []):
        if "pixel_polygon" in dz and dz["pixel_polygon"]:
            pix_poly = dz["pixel_polygon"]
        else:
            pix_poly = [list(projector.meter_to_pixel(p[0], p[1])) for p in dz.get("polygon", [])]
        projected_dwell.append({**dz, "pixel_polygon": pix_poly})

    projected_queue = []
    for qz in zones.get("queue_zones", []):
        if "pixel_polygon" in qz and qz["pixel_polygon"]:
            pix_poly = qz["pixel_polygon"]
        else:
            pix_poly = [list(projector.meter_to_pixel(p[0], p[1])) for p in qz.get("polygon", [])]
        projected_queue.append({**qz, "pixel_polygon": pix_poly})

    projected_tripwires = []
    for tw in zones.get("tripwires", []):
        if "pixel_start" in tw and "pixel_end" in tw and tw["pixel_start"] and tw["pixel_end"]:
            p1 = tw["pixel_start"]
            p2 = tw["pixel_end"]
        else:
            p1 = list(projector.meter_to_pixel(tw["start"][0], tw["start"][1]))
            p2 = list(projector.meter_to_pixel(tw["end"][0], tw["end"][1]))
        projected_tripwires.append({**tw, "pixel_start": p1, "pixel_end": p2})

    return JSONResponse(
        content={
            "homography_calibration": zones.get("homography_calibration", {}),
            "floor_boundary_pixels": floor_boundary_pixels,
            "floor_width_meters": floor_w,
            "floor_height_meters": floor_h,
            "dwell_zones": projected_dwell,
            "queue_zones": projected_queue,
            "tripwires": projected_tripwires,
        }
    )


@app.post("/api/zones/save")
async def save_zones(request: Request) -> JSONResponse:
    """
    Принимает зоны, нарисованные мышкой на виде с камеры (в пикселях [u, v]),
    автоматически конвертирует их в метры через матрицу гомографии H,
    сохраняет в configs/zones.json и мгновенно применяет к работающему пайплайну!
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Неверный JSON") from exc

    projector = metrics_store.get_projector()
    zones_file = "configs/zones.json"

    if os.path.exists(zones_file):
        with open(zones_file, "r", encoding="utf-8") as f:
            current_zones = json.load(f)
    else:
        current_zones = {}

    cfg = metrics_store.get_config()
    floor_w = float(cfg.get("floor_width_meters", 12.0))
    floor_h = float(cfg.get("floor_height_meters", 10.0))

    def clamp_m(x: float, y: float, margin: float = 0.05) -> Tuple[float, float]:
        cx = max(margin, min(floor_w - margin, float(x)))
        cy = max(margin, min(floor_h - margin, float(y)))
        return round(float(cx), 2), round(float(cy), 2)

    # 1. Обновление калибровки гомографии (если переданы 4 опорные точки в пикселях)
    if "homography_calibration" in body and body["homography_calibration"]:
        current_zones["homography_calibration"] = body["homography_calibration"]
        try:
            from src.geometry.homography import HomographyProjector
            h_calib = current_zones["homography_calibration"]
            projector = HomographyProjector(
                src_pixel_points=h_calib["src_pixel_points"],
                dst_meter_points=h_calib["dst_meter_points"],
            )
            metrics_store.set_projector(projector)
        except Exception as e:
            print(f"[save_zones] Ошибка калибровки: {e}")

    if not projector and "homography_calibration" in current_zones:
        try:
            from src.geometry.homography import HomographyProjector
            h_calib = current_zones["homography_calibration"]
            projector = HomographyProjector(
                src_pixel_points=h_calib["src_pixel_points"],
                dst_meter_points=h_calib["dst_meter_points"],
            )
            metrics_store.set_projector(projector)
        except Exception:
            pass

    # 2. Обновление Dwell зон (витрины)
    if "dwell_zones" in body:
        updated_dwell = []
        for dz in body["dwell_zones"]:
            has_pixels = "pixel_polygon" in dz and dz["pixel_polygon"] and projector
            if has_pixels:
                meter_poly = [
                    list(clamp_m(*projector.pixel_to_meter(u, v), margin=0.0))
                    for u, v in dz["pixel_polygon"]
                ]
            else:
                meter_poly = [list(clamp_m(p[0], p[1], margin=0.0)) for p in dz.get("polygon", [])]

            item = {
                "zone_id": dz.get("zone_id", f"dwell_{len(updated_dwell)+1}"),
                "name": dz.get("name", dz.get("zone_id", "Витрина")),
                "threshold_seconds": float(dz.get("threshold_seconds", 1.5)),
                "coord_space": "meter",
                "polygon": meter_poly,
            }
            if "pixel_polygon" in dz and dz["pixel_polygon"]:
                item["pixel_polygon"] = [[int(pt[0]), int(pt[1])] for pt in dz["pixel_polygon"]]
            elif projector:
                item["pixel_polygon"] = [list(projector.meter_to_pixel(p[0], p[1])) for p in meter_poly]

            updated_dwell.append(item)
        current_zones["dwell_zones"] = updated_dwell

    # 3. Обновление зон касс (Queue)
    if "queue_zones" in body:
        updated_queue = []
        for qz in body["queue_zones"]:
            has_pixels = "pixel_polygon" in qz and qz["pixel_polygon"] and projector
            if has_pixels:
                meter_poly = [
                    list(clamp_m(*projector.pixel_to_meter(u, v), margin=0.0))
                    for u, v in qz["pixel_polygon"]
                ]
            else:
                meter_poly = [list(clamp_m(p[0], p[1], margin=0.0)) for p in qz.get("polygon", [])]

            item = {
                "zone_id": qz.get("zone_id", f"queue_{len(updated_queue)+1}"),
                "name": qz.get("name", qz.get("zone_id", "Касса")),
                "max_people_threshold": int(qz.get("max_people_threshold", 2)),
                "coord_space": "meter",
                "polygon": meter_poly,
            }
            if "pixel_polygon" in qz and qz["pixel_polygon"]:
                item["pixel_polygon"] = [[int(pt[0]), int(pt[1])] for pt in qz["pixel_polygon"]]
            elif projector:
                item["pixel_polygon"] = [list(projector.meter_to_pixel(p[0], p[1])) for p in meter_poly]

            updated_queue.append(item)
        current_zones["queue_zones"] = updated_queue

    # 4. Обновление линий Tripwires (вход/выход)
    if "tripwires" in body:
        updated_tw = []
        for tw in body["tripwires"]:
            has_pixels = "pixel_start" in tw and "pixel_end" in tw and projector
            if has_pixels:
                raw_p1 = projector.pixel_to_meter(*tw["pixel_start"])
                raw_p2 = projector.pixel_to_meter(*tw["pixel_end"])
                p1_m = list(clamp_m(raw_p1[0], raw_p1[1], margin=0.1))
                p2_m = list(clamp_m(raw_p2[0], raw_p2[1], margin=0.1))
            else:
                raw_p1 = tw.get("start", [0.5, 5.0])
                raw_p2 = tw.get("end", [11.5, 5.0])
                p1_m = list(clamp_m(raw_p1[0], raw_p1[1], margin=0.1))
                p2_m = list(clamp_m(raw_p2[0], raw_p2[1], margin=0.1))

            # Если точки случайно схлопнулись при клике в одну точку
            if p1_m[0] == p2_m[0] and p1_m[1] == p2_m[1]:
                p2_m[0] = min(floor_w - 0.1, p2_m[0] + 1.0)

            item = {
                "line_id": tw.get("line_id", f"tripwire_{len(updated_tw)+1}"),
                "name": tw.get("name", tw.get("line_id", "Линия входа")),
                "coord_space": "meter",
                "start": p1_m,
                "end": p2_m,
            }
            if "pixel_start" in tw and "pixel_end" in tw and tw["pixel_start"] and tw["pixel_end"]:
                item["pixel_start"] = [int(tw["pixel_start"][0]), int(tw["pixel_start"][1])]
                item["pixel_end"] = [int(tw["pixel_end"][0]), int(tw["pixel_end"][1])]
            elif projector:
                item["pixel_start"] = list(projector.meter_to_pixel(p1_m[0], p1_m[1]))
                item["pixel_end"] = list(projector.meter_to_pixel(p2_m[0], p2_m[1]))

            updated_tw.append(item)
        current_zones["tripwires"] = updated_tw

    with open(zones_file, "w", encoding="utf-8") as f:
        json.dump(current_zones, f, ensure_ascii=False, indent=2)

    # Немедленное обновление кэша зон в памяти
    metrics_store.set_zones_config(current_zones, {"floor_width_meters": floor_w, "floor_height_meters": floor_h})
    metrics_store.signal_zones_updated()

    return JSONResponse(
        content={
            "status": "success",
            "message": "Разметка успешно сохранена и применена к системе!",
            "zones": current_zones,
        }
    )


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            snapshot = metrics_store.get_snapshot()
            await websocket.send_json(snapshot)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <title>SpaceMetrics AI — Retail Analytics & Visual Zone Editor</title>
  <style>
    :root {
      --bg: #090d16;
      --card: #111827;
      --card-hover: #1f2937;
      --border: #1e293b;
      --accent: #00e5ff;
      --accent-glow: rgba(0, 229, 255, 0.15);
      --warn: #f59e0b;
      --danger: #ef4444;
      --ok: #10b981;
      --text: #f1f5f9;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 20px 24px;
      line-height: 1.5;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
    }
    .badge {
      background: var(--accent-glow);
      border: 1px solid var(--accent);
      color: var(--accent);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
    }
    .nav-tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
    }
    .tab-btn {
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 10px 20px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }
    .tab-btn.active {
      background: var(--accent);
      color: #090d16;
      border-color: var(--accent);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    .kpi {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }
    .kpi-title { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-val { font-size: 30px; font-weight: 700; color: var(--accent); margin-top: 4px; }
    .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 10px;
    }
    .card-title { font-size: 16px; font-weight: 600; color: var(--text); }
    .btn-group {
      display: inline-flex;
      background: #0b1120;
      border-radius: 8px;
      padding: 3px;
      border: 1px solid var(--border);
    }
    .btn {
      background: transparent;
      border: none;
      color: var(--muted);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .btn.active { background: var(--accent); color: #0b1120; font-weight: 600; }
    canvas {
      width: 100%;
      background: #020617;
      border-radius: 10px;
      border: 1px solid var(--border);
      display: block;
    }
    .attention-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
    .attention-table th { text-align: left; padding: 8px 10px; color: var(--muted); border-bottom: 1px solid var(--border); }
    .attention-table td { padding: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
    .progress-bar-bg { background: #1e293b; height: 6px; border-radius: 4px; overflow: hidden; margin-top: 4px; }
    .progress-bar-fill { background: linear-gradient(90deg, #00e5ff, #3b82f6); height: 100%; border-radius: 4px; }
    pre {
      background: #020617;
      padding: 12px;
      border-radius: 8px;
      font-size: 12px;
      overflow-x: auto;
      max-height: 250px;
      color: #7dd3fc;
      border: 1px solid var(--border);
      margin-top: 8px;
    }
    .legend-bar { display: flex; align-items: center; justify-content: space-between; margin-top: 10px; font-size: 12px; color: var(--muted); }
    .color-gradient { flex: 1; height: 10px; margin: 0 12px; border-radius: 5px; border: 1px solid #334155; }

    /* Стили для интерактивного редактора зон */
    .editor-grid {
      display: grid;
      grid-template-columns: 2.2fr 1fr;
      gap: 20px;
    }
    .editor-canvas-container {
      position: relative;
      background: #000;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    #cameraCanvas {
      width: 100%;
      height: auto;
      cursor: crosshair;
    }
    .editor-panel {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .form-label {
      font-size: 13px;
      color: var(--muted);
      font-weight: 500;
    }
    .form-input {
      background: #0f172a;
      border: 1px solid var(--border);
      color: #fff;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    .action-btn {
      padding: 10px 14px;
      border-radius: 8px;
      border: none;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s ease;
    }
    .btn-primary { background: #00e5ff; color: #0b1120; }
    .btn-primary:hover { background: #33ecff; }
    .btn-success { background: #10b981; color: #fff; }
    .btn-success:hover { background: #059669; }
    .btn-secondary { background: #1e293b; color: #f1f5f9; border: 1px solid #334155; }
    .btn-secondary:hover { background: #334155; }
    .btn-danger { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    .btn-danger:hover { background: #ef4444; color: #fff; }
    .zone-list-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #0b1120;
      border: 1px solid var(--border);
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 13px;
    }
    .zone-badge {
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
    }
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #10b981;
      color: #fff;
      padding: 12px 20px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      box-shadow: 0 4px 14px rgba(0,0,0,0.4);
      display: none;
      z-index: 999;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SpaceMetrics AI <span style="font-weight:300;color:var(--muted)">| Zero-Video Retail Sensor</span></h1>
      <p style="font-size:13px;color:var(--muted);margin-top:4px;">
        Калибровка гомографии • Интерактивная разметка зон с камеры • ASUS UGen300 (Hailo-10H)
      </p>
    </div>
    <span class="badge">● SENSOR ONLINE</span>
  </header>

  <!-- Вкладки навигации -->
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('analytics')">📊 Аналитика и Тепловая карта</button>
    <button class="tab-btn" onclick="switchTab('live')">🎥 Прямой эфир трекинга (Live View)</button>
    <button class="tab-btn" onclick="switchTab('editor')">✏️ Разметка зон (Вид с камеры)</button>
  </div>

  <!-- ВКЛАДКА 1: АНАЛИТИКА -->
  <div id="tab-analytics" class="tab-content active">
    <div class="kpis">
      <div class="kpi"><div class="kpi-title">Людей на полу</div><div class="kpi-val" id="kpi-people">0</div></div>
      <div class="kpi"><div class="kpi-title">FPS инференса</div><div class="kpi-val" id="kpi-fps">0.0</div></div>
      <div class="kpi"><div class="kpi-title">Трафик Вход / Выход</div><div class="kpi-val" id="kpi-io">0 / 0</div></div>
      <div class="kpi"><div class="kpi-title">Задержек у витрин (Dwell)</div><div class="kpi-val" id="kpi-dwell">0</div></div>
    </div>

    <div class="main-grid">
      <div class="card">
        <div class="card-header">
          <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
            <div class="card-title">Визуализация пространства</div>
            <div class="btn-group" id="viewToggleGroup">
              <button class="btn active" onclick="setMainViewMode('floor')">🗺️ 2D План</button>
              <button class="btn" onclick="setMainViewMode('live')">🎥 Live Камера</button>
              <button class="btn" onclick="setMainViewMode('split')">🔲 Сплит</button>
            </div>
          </div>
          <div style="display: flex; gap: 8px; align-items: center;" id="heatmapControls">
            <div class="btn-group" id="modeGroup">
              <button class="btn active" onclick="setHeatmapMode('total')">Общее</button>
              <button class="btn" onclick="setHeatmapMode('dwell')">Внимание (Dwell)</button>
              <button class="btn" onclick="setHeatmapMode('flow')">Транзит (Flow)</button>
            </div>
            <div class="btn-group" id="colormapGroup">
              <button class="btn active" onclick="setColormap('turbo')">Turbo</button>
              <button class="btn" onclick="setColormap('jet')">Jet</button>
              <button class="btn" onclick="setColormap('inferno')">Inferno</button>
            </div>
          </div>
        </div>

        <div id="mainViewContainer" style="display: flex; gap: 14px; width: 100%;">
          <div id="floorCanvasWrapper" style="flex: 1; min-width: 0;">
            <canvas id="floorCanvas" width="720" height="600" style="width: 100%; height: auto;"></canvas>
            <div class="legend-bar" id="legendContainer">
              <span>0 сек (Холодно)</span>
              <div class="color-gradient" id="legendGradient"></div>
              <span>Пик присутствия (Горячо)</span>
            </div>
          </div>
          <div id="liveStreamWrapper" style="flex: 1; min-width: 0; display: none; background: #000; border-radius: 10px; overflow: hidden; border: 1px solid var(--border); position: relative;">
            <img id="liveStreamAnalytics" src="/api/camera/stream" style="width: 100%; height: auto; display: block;" />
            <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(0,0,0,0.7); padding: 4px 10px; border-radius: 6px; font-size: 11px; color: #10b981; font-weight: 600;">
              ● LIVE VIEW (Zero-Video Stream)
            </div>
          </div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 24px;">
        <div class="card">
          <div class="card-header">
            <div class="card-title">Доля внимания к витринам (Attention Share)</div>
          </div>
          <table class="attention-table">
            <thead>
              <tr>
                <th>Зона / Витрина</th>
                <th style="text-align: right;">Время</th>
                <th style="text-align: right; width: 35%;">Доля (%)</th>
              </tr>
            </thead>
            <tbody id="attentionBody">
              <tr><td colspan="3" style="color:var(--muted)">Сбор аналитики...</td></tr>
            </tbody>
          </table>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">Подсчет трафика по линиям входа (Tripwires)</div>
          </div>
          <table class="attention-table">
            <thead>
              <tr>
                <th>Линия / Вход</th>
                <th style="text-align: right;">Вход</th>
                <th style="text-align: right;">Выход</th>
                <th style="text-align: right;">Всего</th>
              </tr>
            </thead>
            <tbody id="tripwiresBody">
              <tr><td colspan="4" style="color:var(--muted)">Ожидание пересечений...</td></tr>
            </tbody>
          </table>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">Обезличенный JSON-стрим (1–2 КБ)</div>
          </div>
          <pre id="jsonLog">Ожидание подключения к WebSocket...</pre>
        </div>
      </div>
    </div>
  </div>

  <!-- ВКЛАДКА 2: ПРЯМОЙ ЭФИР ТРЕКИНГА (LIVE VIEW) -->
  <div id="tab-live" class="tab-content">
    <div style="display: flex; flex-direction: column; gap: 16px;">
      <!-- Панель управления слоями визуализации -->
      <div class="card" style="padding: 14px 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
          <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
            <span style="font-weight: 600; font-size: 13px; color: var(--text);">Слои трекинга на видео:</span>
            <label style="display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; color: #7dd3fc;">
              <input type="checkbox" id="chkBoxes" checked onchange="updateOverlaySettings()" /> Рамки и #ID
            </label>
            <label style="display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; color: #a7f3d0;">
              <input type="checkbox" id="chkTrails" checked onchange="updateOverlaySettings()" /> Траектории
            </label>
            <label style="display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; color: #fde68a;">
              <input type="checkbox" id="chkAnchors" checked onchange="updateOverlaySettings()" /> Точки стоп (м)
            </label>
            <label style="display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; color: #c4b5fd;">
              <input type="checkbox" id="chkZones" checked onchange="updateOverlaySettings()" /> Зоны и Линии
            </label>
            <label style="display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; color: var(--muted);">
              <input type="checkbox" id="chkHud" checked onchange="updateOverlaySettings()" /> Инфо HUD
            </label>
          </div>
          <div style="display: flex; gap: 8px;">
            <button class="action-btn btn-secondary" onclick="toggleStreamPause()" id="btnPauseStream">⏸ Пауза</button>
            <button class="action-btn btn-primary" onclick="snapshotLiveFrame()">📸 Снимок кадра</button>
          </div>
        </div>
      </div>

      <!-- Главное окно Live View и таблица активных треков -->
      <div style="display: grid; grid-template-columns: 2.2fr 1fr; gap: 20px;">
        <div class="card" style="padding: 10px; background: #000; border-radius: 12px; position: relative;">
          <img id="liveStreamMain" src="/api/camera/stream" style="width: 100%; height: auto; border-radius: 8px; display: block;" />
          <div style="position: absolute; top: 20px; left: 20px; background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #10b981; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10b981;"></span>
            LIVE STREAM • ZERO-VIDEO EDGE
          </div>
        </div>

        <div class="card" style="display: flex; flex-direction: column; gap: 12px;">
          <div class="card-title">Инспектор треков в реальном времени</div>
          <p style="font-size: 12px; color: var(--muted);">
            Данные людей, обнаруженных прямо сейчас на видеопотоке:
          </p>
          <div style="overflow-y: auto; max-height: 480px;">
            <table class="attention-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Пол (X, Y)</th>
                  <th>Conf</th>
                  <th>Статус</th>
                </tr>
              </thead>
              <tbody id="liveTracksBody">
                <tr><td colspan="4" style="color:var(--muted)">Ожидание детекций...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ВКЛАДКА 3: ИНТЕРАКТИВНЫЙ РЕДАКТОР ЗОН С КАМЕРЫ -->
  <div id="tab-editor" class="tab-content">
    <div class="editor-grid">
      <div class="editor-canvas-container">
        <canvas id="cameraCanvas" width="1280" height="720"></canvas>
        <div style="position: absolute; bottom: 12px; left: 16px; background: rgba(0,0,0,0.7); padding: 4px 10px; border-radius: 6px; font-size: 12px; color: var(--accent);" id="mouseCoords">
          Кликните мышкой на кадре для добавления точек зоны
        </div>
      </div>

      <div class="card editor-panel">
        <div class="card-title">Инструменты черчения мышкой</div>
        
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px 12px; font-size: 12px; color: #a7f3d0; line-height: 1.45;">
          💡 <strong>Как ставить линию входа:</strong> Выберите «Линия», кликните 1-ю точку (старт), затем 2-ю точку (конец) поперек пути движения людей. Зеленая стрелка укажет сторону <strong>ВХОДА</strong>. Линия автоматически привяжется к полу.
        </div>

        <div class="form-group">
          <label class="form-label">Тип создаваемой зоны:</label>
          <div class="btn-group" style="width: 100%;">
            <button class="btn active" style="flex:1" onclick="setDrawingType('dwell')">Витрина</button>
            <button class="btn" style="flex:1" onclick="setDrawingType('queue')">Касса</button>
            <button class="btn" style="flex:1" onclick="setDrawingType('tripwire')">Линия</button>
          </div>
        </div>

        <div class="form-group">
          <label class="form-label">Название зоны:</label>
          <input type="text" id="zoneNameInput" class="form-input" placeholder="Например: Витрина ROG или Касса 1" value="Новая витрина" />
        </div>

        <div class="form-group" id="dwellThresholdGroup">
          <label class="form-label">Порог задержки (сек):</label>
          <input type="number" id="zoneThresholdInput" class="form-input" value="1.5" step="0.5" min="0.5" max="30" />
        </div>

        <div style="display: flex; gap: 8px;">
          <button class="action-btn btn-primary" style="flex: 1;" onclick="finishCurrentZone()">✓ Замкнуть зону</button>
          <button class="action-btn btn-secondary" onclick="undoLastPoint()">↩ Отмена</button>
          <button class="action-btn btn-danger" onclick="clearCurrentDrawing()">✕</button>
        </div>

        <div style="border-top: 1px solid var(--border); margin: 8px 0;"></div>

        <div class="form-label" style="display: flex; justify-content: space-between;">
          <span>Список активных зон:</span>
          <span id="zoneCountBadge" style="color:var(--accent);">0</span>
        </div>

        <div id="zonesListContainer" style="display: flex; flex-direction: column; gap: 8px; max-height: 220px; overflow-y: auto;">
          <!-- Карточки зон -->
        </div>

        <button class="action-btn btn-success" style="margin-top: 10px;" onclick="saveZonesToServer()">
          💾 Сохранить и применить в систему
        </button>
        <button class="action-btn btn-secondary" onclick="reloadCameraFrame()">
          🔄 Обновить стоп-кадр камеры
        </button>
      </div>
    </div>
  </div>

  <div id="toast" class="toast">Зоны сохранены!</div>

  <script>
    // === 1. ПЕРЕКЛЮЧЕНИЕ ВКЛАДОК ===
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      if (tabId === 'analytics') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('tab-analytics').classList.add('active');
        fetch('/api/config').then(r => r.json()).then(data => { if (data.floor_width_meters) { cfg = data; if (latestData) drawFloor(latestData); } });
      } else if (tabId === 'live') {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('tab-live').classList.add('active');
      } else {
        document.querySelectorAll('.tab-btn')[2].classList.add('active');
        document.getElementById('tab-editor').classList.add('active');
        initCameraEditor();
      }
    }

    // === 2. ДАШБОРД АНАЛИТИКИ ===
    const floorCanvas = document.getElementById('floorCanvas');
    const floorCtx = floorCanvas.getContext('2d');
    let currentMode = 'total';
    let currentColormap = 'turbo';
    let cfg = { floor_width_meters: 12.0, floor_height_meters: 10.0, zones: {} };
    let latestData = null;

    fetch('/api/config').then(r => r.json()).then(data => { if (data.floor_width_meters) cfg = data; });

    function setHeatmapMode(mode) {
      currentMode = mode;
      document.querySelectorAll('#modeGroup .btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      if (latestData) drawFloor(latestData);
    }

    function setColormap(cmap) {
      currentColormap = cmap;
      document.querySelectorAll('#colormapGroup .btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      updateLegendGradient();
      if (latestData) drawFloor(latestData);
    }

    function updateLegendGradient() {
      const g = document.getElementById('legendGradient');
      if (currentColormap === 'jet') {
        g.style.background = 'linear-gradient(90deg, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)';
      } else if (currentColormap === 'inferno') {
        g.style.background = 'linear-gradient(90deg, #000004, #57106e, #bb3754, #f98e09, #fcffa4)';
      } else {
        g.style.background = 'linear-gradient(90deg, #30123b, #4582ec, #1ae4b6, #a2fc3c, #fb8022, #7a0403)';
      }
    }
    updateLegendGradient();

    function m2px(xm, ym) {
      return [
        (xm / cfg.floor_width_meters) * floorCanvas.width,
        (ym / cfg.floor_height_meters) * floorCanvas.height
      ];
    }

    function getHeatColor(v) {
      if (currentColormap === 'jet') {
        const hue = (1.0 - v) * 240;
        return `hsla(${hue}, 100%, 50%, ${Math.min(0.75, v * 0.85)})`;
      } else if (currentColormap === 'inferno') {
        const r = Math.min(255, Math.floor(v * 290));
        const g = Math.min(255, Math.floor(Math.pow(v, 1.5) * 230));
        const b = Math.min(255, Math.floor(Math.pow(v, 3.0) * 160));
        return `rgba(${r}, ${g}, ${b}, ${Math.min(0.8, v * 0.9)})`;
      } else {
        const hue = (1.0 - v) * 260 + 20;
        return `hsla(${hue}, 95%, ${40 + v * 20}%, ${Math.min(0.75, v * 0.85)})`;
      }
    }

    function drawFloor(data) {
      floorCtx.clearRect(0, 0, floorCanvas.width, floorCanvas.height);

      const matrix = (data.heatmaps && data.heatmaps[currentMode]) || data.heatmap || [];
      if (matrix.length > 0) {
        const H = matrix.length;
        const W = matrix[0].length;
        const cellW = floorCanvas.width / W;
        const cellH = floorCanvas.height / H;

        for (let r = 0; r < H; r++) {
          for (let c = 0; c < W; c++) {
            const v = matrix[r][c];
            if (v > 0.015) {
              floorCtx.fillStyle = getHeatColor(v);
              floorCtx.fillRect(c * cellW, r * cellH, cellW + 1.2, cellH + 1.2);
            }
          }
        }
      }

      floorCtx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
      floorCtx.lineWidth = 1;
      for (let x = 0; x <= cfg.floor_width_meters; x++) {
        const [px, _] = m2px(x, 0);
        floorCtx.beginPath(); floorCtx.moveTo(px, 0); floorCtx.lineTo(px, floorCanvas.height); floorCtx.stroke();
      }
      for (let y = 0; y <= cfg.floor_height_meters; y++) {
        const [_, py] = m2px(0, y);
        floorCtx.beginPath(); floorCtx.moveTo(0, py); floorCtx.lineTo(floorCanvas.width, py); floorCtx.stroke();
      }

      const dwellZones = (cfg.zones && cfg.zones.dwell_zones) || [];
      dwellZones.forEach(z => {
        floorCtx.strokeStyle = 'rgba(0, 229, 255, 0.7)';
        floorCtx.fillStyle = 'rgba(0, 229, 255, 0.04)';
        floorCtx.lineWidth = 2;
        floorCtx.beginPath();
        (z.polygon || []).forEach((pt, idx) => {
          const [px, py] = m2px(pt[0], pt[1]);
          if (idx === 0) floorCtx.moveTo(px, py); else floorCtx.lineTo(px, py);
        });
        floorCtx.closePath();
        floorCtx.fill();
        floorCtx.stroke();

        if (z.polygon && z.polygon.length > 0) {
          const [p0x, p0y] = m2px(z.polygon[0][0], z.polygon[0][1]);
          floorCtx.fillStyle = '#00e5ff';
          floorCtx.font = '11px sans-serif';
          floorCtx.fillText(z.name || z.zone_id, p0x + 8, p0y + 18);
        }
      });

      const queueZones = (cfg.zones && cfg.zones.queue_zones) || [];
      queueZones.forEach(qz => {
        floorCtx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        floorCtx.fillStyle = 'rgba(239, 68, 68, 0.04)';
        floorCtx.lineWidth = 2;
        floorCtx.beginPath();
        (qz.polygon || []).forEach((pt, idx) => {
          const [px, py] = m2px(pt[0], pt[1]);
          if (idx === 0) floorCtx.moveTo(px, py); else floorCtx.lineTo(px, py);
        });
        floorCtx.closePath();
        floorCtx.fill();
        floorCtx.stroke();

        if (qz.polygon && qz.polygon.length > 0) {
          const [p0x, p0y] = m2px(qz.polygon[0][0], qz.polygon[0][1]);
          floorCtx.fillStyle = '#ef4444';
          floorCtx.font = '11px sans-serif';
          floorCtx.fillText(qz.name || qz.zone_id, p0x + 8, p0y + 18);
        }
      });

      const tripwires = (cfg.zones && cfg.zones.tripwires) || [];
      const twDataMap = (data && data.tripwires) || {};
      tripwires.forEach(tw => {
        const [x1, y1] = m2px(tw.start[0], tw.start[1]);
        const [x2, y2] = m2px(tw.end[0], tw.end[1]);
        const stat = twDataMap[tw.line_id] || { in_count: 0, out_count: 0 };

        // Линия с подсветкой
        floorCtx.save();
        floorCtx.shadowBlur = 6;
        floorCtx.shadowColor = '#10b981';
        floorCtx.strokeStyle = '#10b981';
        floorCtx.lineWidth = 3;
        floorCtx.beginPath();
        floorCtx.moveTo(x1, y1); floorCtx.lineTo(x2, y2);
        floorCtx.stroke();
        floorCtx.restore();

        // Точки на концах отрезка
        floorCtx.fillStyle = '#10b981';
        floorCtx.beginPath(); floorCtx.arc(x1, y1, 5, 0, Math.PI * 2); floorCtx.fill();
        floorCtx.beginPath(); floorCtx.arc(x2, y2, 5, 0, Math.PI * 2); floorCtx.fill();

        // Перпендикулярная стрелка направления входа
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len > 8) {
          const nx = -dy / len;
          const ny = dx / len;
          const arrowLen = 14;
          const ax = mx + nx * arrowLen;
          const ay = my + ny * arrowLen;
          floorCtx.strokeStyle = '#34d399';
          floorCtx.lineWidth = 2;
          floorCtx.beginPath();
          floorCtx.moveTo(mx, my); floorCtx.lineTo(ax, ay);
          floorCtx.stroke();
          const angle = Math.atan2(ny, nx);
          floorCtx.fillStyle = '#34d399';
          floorCtx.beginPath();
          floorCtx.moveTo(ax, ay);
          floorCtx.lineTo(ax - 6 * Math.cos(angle - Math.PI / 6), ay - 6 * Math.sin(angle - Math.PI / 6));
          floorCtx.lineTo(ax - 6 * Math.cos(angle + Math.PI / 6), ay - 6 * Math.sin(angle + Math.PI / 6));
          floorCtx.closePath();
          floorCtx.fill();
        }

        // Подпись с актуальным счетчиком
        floorCtx.fillStyle = '#34d399';
        floorCtx.font = 'bold 11px sans-serif';
        const label = `${tw.name || tw.line_id} (Вх: ${stat.in_count || 0} | Вых: ${stat.out_count || 0})`;
        floorCtx.fillText(label, mx + 8, my - 6);
      });

      (data.floor_tracks || []).forEach(t => {
        const [px, py] = m2px(t.x_m, t.y_m);
        floorCtx.fillStyle = '#10b981';
        floorCtx.beginPath();
        floorCtx.arc(px, py, 6, 0, Math.PI * 2);
        floorCtx.fill();
        floorCtx.strokeStyle = '#ffffff';
        floorCtx.lineWidth = 1.5;
        floorCtx.stroke();
        floorCtx.fillStyle = '#ffffff';
        floorCtx.font = '11px sans-serif';
        floorCtx.fillText(`ID:${t.track_id} (${t.x_m.toFixed(1)}m, ${t.y_m.toFixed(1)}m)`, px + 8, py - 4);
      });
    }

    function renderAttentionTable(attention) {
      const tbody = document.getElementById('attentionBody');
      const items = Object.values(attention || {});
      if (items.length === 0) return;

      tbody.innerHTML = items.map(item => `
        <tr>
          <td><strong>${item.name || item.zone_id}</strong></td>
          <td style="text-align: right; color: var(--accent);">${item.accumulated_seconds.toFixed(1)}s</td>
          <td style="text-align: right;">
            <div><strong>${item.attention_share_percent}%</strong></div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" style="width: ${item.attention_share_percent}%"></div>
            </div>
          </td>
        </tr>
      `).join('');
    }

    function renderTripwireTable(tripwires) {
      const tbody = document.getElementById('tripwiresBody');
      if (!tbody) return;
      const items = Object.values(tripwires || {});
      if (items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="color:var(--muted)">Линии не настроены или ожидают событий</td></tr>';
        return;
      }
      tbody.innerHTML = items.map(tw => `
        <tr>
          <td>
            <div style="display:flex;align-items:center;gap:6px;">
              <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;"></span>
              <strong>${tw.name || tw.line_id}</strong>
            </div>
          </td>
          <td style="text-align: right; color: #10b981; font-weight:600;">+${tw.in_count || 0}</td>
          <td style="text-align: right; color: #f59e0b; font-weight:600;">-${tw.out_count || 0}</td>
          <td style="text-align: right; color: var(--accent); font-weight:700;">${tw.total_crossings || (tw.in_count + tw.out_count) || 0}</td>
        </tr>
      `).join('');
    }

    function setMainViewMode(mode) {
      document.querySelectorAll('#viewToggleGroup .btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      const floorWrap = document.getElementById('floorCanvasWrapper');
      const liveWrap = document.getElementById('liveStreamWrapper');
      const heatmapControls = document.getElementById('heatmapControls');
      if (mode === 'floor') {
        floorWrap.style.display = 'block';
        liveWrap.style.display = 'none';
        if (heatmapControls) heatmapControls.style.display = 'flex';
      } else if (mode === 'live') {
        floorWrap.style.display = 'none';
        liveWrap.style.display = 'block';
        if (heatmapControls) heatmapControls.style.display = 'none';
      } else { // split
        floorWrap.style.display = 'block';
        liveWrap.style.display = 'block';
        if (heatmapControls) heatmapControls.style.display = 'flex';
      }
    }

    let isStreamPaused = false;
    function toggleStreamPause() {
      isStreamPaused = !isStreamPaused;
      const btn = document.getElementById('btnPauseStream');
      const imgMain = document.getElementById('liveStreamMain');
      const imgAna = document.getElementById('liveStreamAnalytics');
      if (isStreamPaused) {
        btn.textContent = '▶ Возобновить';
        btn.className = 'action-btn btn-success';
        if (imgMain) imgMain.src = '/api/camera/live_frame?t=' + Date.now();
        if (imgAna) imgAna.src = '/api/camera/live_frame?t=' + Date.now();
      } else {
        btn.textContent = '⏸ Пауза';
        btn.className = 'action-btn btn-secondary';
        if (imgMain) imgMain.src = '/api/camera/stream';
        if (imgAna) imgAna.src = '/api/camera/stream';
      }
    }

    function snapshotLiveFrame() {
      const a = document.createElement('a');
      a.href = '/api/camera/live_frame?download=1&t=' + Date.now();
      a.download = `spacemetrics_live_${Date.now()}.jpg`;
      a.target = '_blank';
      a.click();
    }

    function updateOverlaySettings() {
      const payload = {
        boxes: document.getElementById('chkBoxes').checked,
        trails: document.getElementById('chkTrails').checked,
        anchors: document.getElementById('chkAnchors').checked,
        zones: document.getElementById('chkZones').checked,
        hud: document.getElementById('chkHud').checked,
      };
      fetch('/api/camera/overlay_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    }

    function renderLiveTracksTable(floorTracks) {
      const tbody = document.getElementById('liveTracksBody');
      if (!tbody) return;
      if (!floorTracks || floorTracks.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="color:var(--muted)">В кадре нет активных людей</td></tr>';
        return;
      }
      tbody.innerHTML = floorTracks.map(t => `
        <tr>
          <td><strong style="color:var(--accent);">#${t.track_id}</strong></td>
          <td style="color:#fde68a;">${t.x_m.toFixed(1)}м, ${t.y_m.toFixed(1)}м</td>
          <td style="color:#a7f3d0;">${Math.round(t.conf * 100)}%</td>
          <td><span class="badge" style="padding:2px 8px; font-size:10px;">В кадре</span></td>
        </tr>
      `).join('');
    }

    const ws = new WebSocket(`ws://${location.host}/ws/metrics`);
    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      latestData = data;

      document.getElementById('kpi-people').textContent = data.active_people_count || 0;
      document.getElementById('kpi-fps').textContent = (data.fps || 0).toFixed(1);
      
      let totalIn = 0, totalOut = 0;
      Object.values(data.tripwires || {}).forEach(tw => {
        totalIn += (tw.in_count || 0);
        totalOut += (tw.out_count || 0);
      });
      document.getElementById('kpi-io').textContent = `${totalIn} / ${totalOut}`;
      document.getElementById('kpi-dwell').textContent = data.total_dwell_events || 0;

      if (data.zone_attention) renderAttentionTable(data.zone_attention);
      renderTripwireTable(data.tripwires);
      renderLiveTracksTable(data.floor_tracks);

      const logCopy = Object.assign({}, data);
      delete logCopy.heatmaps;
      delete logCopy.heatmap;
      delete logCopy.active_heatmap;
      document.getElementById('jsonLog').textContent = JSON.stringify(logCopy, null, 2);

      drawFloor(data);
    };

    // === 3. ИНТЕРАКТИВНЫЙ РЕДАКТОР ЗОН С КАМЕРЫ ===
    const cameraCanvas = document.getElementById('cameraCanvas');
    const camCtx = cameraCanvas.getContext('2d');
    let camImage = new Image();
    let currentDrawingType = 'dwell'; // dwell, queue, tripwire
    let currentPoints = [];
    let mousePos = { x: 0, y: 0 };
    let editorZones = { dwell_zones: [], queue_zones: [], tripwires: [] };
    let floorBoundaryPixels = [];

    function loadEditorZones(cb) {
      fetch('/api/zones/camera_projected').then(r => r.json()).then(data => {
        floorBoundaryPixels = data.floor_boundary_pixels || [];
        editorZones = {
          dwell_zones: data.dwell_zones || [],
          queue_zones: data.queue_zones || [],
          tripwires: data.tripwires || [],
        };
        renderZonesList();
        redrawCameraCanvas();
        if (cb) cb();
      });
    }
    loadEditorZones();

    function initCameraEditor() {
      reloadCameraFrame();
      loadEditorZones();
    }

    function reloadCameraFrame() {
      camImage.src = '/api/camera/frame?t=' + Date.now();
      camImage.onload = () => {
        cameraCanvas.width = camImage.width || 1280;
        cameraCanvas.height = camImage.height || 720;
        redrawCameraCanvas();
      };
    }

    function setDrawingType(type) {
      currentDrawingType = type;
      currentPoints = [];
      document.querySelectorAll('#tab-editor .btn-group .btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      const titles = { dwell: 'Витрина', queue: 'Касса', tripwire: 'Линия входа' };
      document.getElementById('zoneNameInput').value = `${titles[type]} #${getZoneNextIndex(type)}`;
      document.getElementById('dwellThresholdGroup').style.display = (type === 'dwell') ? 'flex' : 'none';
      redrawCameraCanvas();
    }

    function getZoneNextIndex(type) {
      if (type === 'dwell') return (editorZones.dwell_zones.length + 1);
      if (type === 'queue') return (editorZones.queue_zones.length + 1);
      return (editorZones.tripwires.length + 1);
    }

    cameraCanvas.addEventListener('mousemove', (e) => {
      const rect = cameraCanvas.getBoundingClientRect();
      const scaleX = cameraCanvas.width / rect.width;
      const scaleY = cameraCanvas.height / rect.height;
      mousePos.x = Math.round((e.clientX - rect.left) * scaleX);
      mousePos.y = Math.round((e.clientY - rect.top) * scaleY);
      document.getElementById('mouseCoords').textContent = `Курсор: X: ${mousePos.x}px, Y: ${mousePos.y}px | Точек в контуре: ${currentPoints.length}`;
      if (currentPoints.length > 0) redrawCameraCanvas();
    });

    cameraCanvas.addEventListener('click', (e) => {
      const rect = cameraCanvas.getBoundingClientRect();
      const scaleX = cameraCanvas.width / rect.width;
      const scaleY = cameraCanvas.height / rect.height;
      const px = Math.round((e.clientX - rect.left) * scaleX);
      const py = Math.round((e.clientY - rect.top) * scaleY);

      currentPoints.push([px, py]);

      if (currentDrawingType === 'tripwire' && currentPoints.length === 2) {
        finishCurrentZone();
      } else {
        redrawCameraCanvas();
      }
    });

    cameraCanvas.addEventListener('dblclick', () => {
      finishCurrentZone();
    });

    function undoLastPoint() {
      if (currentPoints.length > 0) {
        currentPoints.pop();
        redrawCameraCanvas();
      }
    }

    function clearCurrentDrawing() {
      currentPoints = [];
      redrawCameraCanvas();
    }

    function finishCurrentZone() {
      const name = document.getElementById('zoneNameInput').value.trim() || 'Новая зона';
      if (currentDrawingType === 'tripwire') {
        if (currentPoints.length < 2) { alert('Для линии входа кликните 2 точки (начало и конец)!'); return; }
        const newId = `tripwire_${Date.now() % 10000}`;
        editorZones.tripwires.push({
          line_id: newId,
          name: name,
          pixel_start: currentPoints[0],
          pixel_end: currentPoints[1]
        });
        currentPoints = [];
        document.getElementById('zoneNameInput').value = `Линия входа #${editorZones.tripwires.length + 1}`;
        renderZonesList();
        redrawCameraCanvas();
        return;
      } else if (currentDrawingType === 'queue') {
        if (currentPoints.length < 3) { alert('Для зоны касс кликните минимум 3 точки!'); return; }
        editorZones.queue_zones.push({
          zone_id: `queue_${Date.now() % 10000}`,
          name: name,
          max_people_threshold: 2,
          pixel_polygon: [...currentPoints]
        });
        document.getElementById('zoneNameInput').value = `Касса #${editorZones.queue_zones.length + 1}`;
      } else {
        if (currentPoints.length < 3) { alert('Для витрины кликните минимум 3 точки по углам!'); return; }
        const thresh = parseFloat(document.getElementById('zoneThresholdInput').value) || 1.5;
        editorZones.dwell_zones.push({
          zone_id: `dwell_${Date.now() % 10000}`,
          name: name,
          threshold_seconds: thresh,
          pixel_polygon: [...currentPoints]
        });
        document.getElementById('zoneNameInput').value = `Витрина #${editorZones.dwell_zones.length + 1}`;
      }

      currentPoints = [];
      renderZonesList();
      redrawCameraCanvas();
    }

    function deleteZone(type, index) {
      if (type === 'dwell') editorZones.dwell_zones.splice(index, 1);
      else if (type === 'queue') editorZones.queue_zones.splice(index, 1);
      else if (type === 'tripwire') editorZones.tripwires.splice(index, 1);
      renderZonesList();
      redrawCameraCanvas();
    }

    function renderZonesList() {
      const container = document.getElementById('zonesListContainer');
      container.innerHTML = '';
      let total = 0;

      (editorZones.dwell_zones || []).forEach((z, i) => {
        total++;
        const item = document.createElement('div');
        item.className = 'zone-list-item';
        item.innerHTML = `
          <div>
            <span class="zone-badge" style="background:rgba(0,229,255,0.2);color:#00e5ff;">Витрина</span>
            <strong>${z.name || z.zone_id}</strong> (${(z.pixel_polygon || []).length} точек, ${z.threshold_seconds || 1.5}с)
          </div>
          <button class="action-btn btn-danger" style="padding:4px 8px;font-size:11px;" onclick="deleteZone('dwell', ${i})">🗑</button>
        `;
        container.appendChild(item);
      });

      (editorZones.queue_zones || []).forEach((z, i) => {
        total++;
        const item = document.createElement('div');
        item.className = 'zone-list-item';
        item.innerHTML = `
          <div>
            <span class="zone-badge" style="background:rgba(239,68,68,0.2);color:#ef4444;">Касса</span>
            <strong>${z.name || z.zone_id}</strong> (${(z.pixel_polygon || []).length} точек)
          </div>
          <button class="action-btn btn-danger" style="padding:4px 8px;font-size:11px;" onclick="deleteZone('queue', ${i})">🗑</button>
        `;
        container.appendChild(item);
      });

      (editorZones.tripwires || []).forEach((tw, i) => {
        total++;
        const item = document.createElement('div');
        item.className = 'zone-list-item';
        item.innerHTML = `
          <div>
            <span class="zone-badge" style="background:rgba(16,185,129,0.2);color:#10b981;">Линия</span>
            <strong>${tw.name || tw.line_id}</strong>
          </div>
          <button class="action-btn btn-danger" style="padding:4px 8px;font-size:11px;" onclick="deleteZone('tripwire', ${i})">🗑</button>
        `;
        container.appendChild(item);
      });

      document.getElementById('zoneCountBadge').textContent = total;
    }

    function redrawCameraCanvas() {
      camCtx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
      if (camImage.complete && camImage.naturalWidth > 0) {
        camCtx.drawImage(camImage, 0, 0, cameraCanvas.width, cameraCanvas.height);
      }

      // 1. Отображение калибровочного периметра пола (12х10м)
      if (floorBoundaryPixels && floorBoundaryPixels.length === 4) {
        camCtx.save();
        camCtx.setLineDash([8, 6]);
        camCtx.strokeStyle = 'rgba(0, 229, 255, 0.45)';
        camCtx.lineWidth = 2;
        camCtx.beginPath();
        floorBoundaryPixels.forEach((p, idx) => {
          if (idx === 0) camCtx.moveTo(p[0], p[1]); else camCtx.lineTo(p[0], p[1]);
        });
        camCtx.closePath();
        camCtx.stroke();
        camCtx.fillStyle = 'rgba(0, 229, 255, 0.03)';
        camCtx.fill();
        camCtx.fillStyle = '#00e5ff';
        camCtx.font = '12px sans-serif';
        camCtx.fillText('⬚ Рабочая зона калибровки пола (12м × 10м)', floorBoundaryPixels[0][0] + 10, floorBoundaryPixels[0][1] + 20);
        camCtx.restore();
      }

      // 2. Dwell-зоны (витрины)
      (editorZones.dwell_zones || []).forEach(z => {
        const poly = z.pixel_polygon || [];
        if (poly.length < 2) return;
        camCtx.strokeStyle = '#00e5ff';
        camCtx.fillStyle = 'rgba(0, 229, 255, 0.15)';
        camCtx.lineWidth = 2.5;
        camCtx.beginPath();
        poly.forEach((p, idx) => {
          if (idx === 0) camCtx.moveTo(p[0], p[1]); else camCtx.lineTo(p[0], p[1]);
        });
        camCtx.closePath();
        camCtx.fill();
        camCtx.stroke();

        camCtx.fillStyle = '#00e5ff';
        camCtx.font = '14px sans-serif';
        camCtx.fillText(`★ ${z.name || z.zone_id} (${z.threshold_seconds || 1.5}s)`, poly[0][0] + 6, poly[0][1] - 8);
      });

      // 3. Зоны очередей (кассы)
      (editorZones.queue_zones || []).forEach(qz => {
        const poly = qz.pixel_polygon || [];
        if (poly.length < 2) return;
        camCtx.strokeStyle = '#ef4444';
        camCtx.fillStyle = 'rgba(239, 68, 68, 0.15)';
        camCtx.lineWidth = 2.5;
        camCtx.beginPath();
        poly.forEach((p, idx) => {
          if (idx === 0) camCtx.moveTo(p[0], p[1]); else camCtx.lineTo(p[0], p[1]);
        });
        camCtx.closePath();
        camCtx.fill();
        camCtx.stroke();

        camCtx.fillStyle = '#ef4444';
        camCtx.font = '14px sans-serif';
        camCtx.fillText(`🛒 ${qz.name || qz.zone_id}`, poly[0][0] + 6, poly[0][1] - 8);
      });

      // 4. Линии входа (Tripwires)
      (editorZones.tripwires || []).forEach(tw => {
        if (!tw.pixel_start || !tw.pixel_end) return;
        const [x1, y1] = tw.pixel_start;
        const [x2, y2] = tw.pixel_end;

        camCtx.save();
        camCtx.shadowBlur = 6;
        camCtx.shadowColor = '#10b981';
        camCtx.strokeStyle = '#10b981';
        camCtx.lineWidth = 3.5;
        camCtx.beginPath();
        camCtx.moveTo(x1, y1); camCtx.lineTo(x2, y2);
        camCtx.stroke();
        camCtx.restore();

        // Точки начала (1) и конца (2)
        camCtx.fillStyle = '#10b981';
        camCtx.beginPath(); camCtx.arc(x1, y1, 6, 0, Math.PI * 2); camCtx.fill();
        camCtx.beginPath(); camCtx.arc(x2, y2, 6, 0, Math.PI * 2); camCtx.fill();

        camCtx.fillStyle = '#ffffff';
        camCtx.font = 'bold 11px sans-serif';
        camCtx.fillText('1', x1 - 3, y1 - 9);
        camCtx.fillText('2', x2 - 3, y2 - 9);

        // Стрелка направления входа
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len > 10) {
          const nx = -dy / len;
          const ny = dx / len;
          const ax = mx + nx * 22;
          const ay = my + ny * 22;
          camCtx.strokeStyle = '#34d399';
          camCtx.lineWidth = 2;
          camCtx.beginPath();
          camCtx.moveTo(mx, my); camCtx.lineTo(ax, ay);
          camCtx.stroke();

          const angle = Math.atan2(ny, nx);
          camCtx.fillStyle = '#34d399';
          camCtx.beginPath();
          camCtx.moveTo(ax, ay);
          camCtx.lineTo(ax - 8 * Math.cos(angle - Math.PI / 6), ay - 8 * Math.sin(angle - Math.PI / 6));
          camCtx.lineTo(ax - 8 * Math.cos(angle + Math.PI / 6), ay - 8 * Math.sin(angle + Math.PI / 6));
          camCtx.closePath();
          camCtx.fill();

          camCtx.fillStyle = '#34d399';
          camCtx.font = 'bold 12px sans-serif';
          camCtx.fillText('ВХОД →', ax + 6, ay + 4);
        }

        camCtx.fillStyle = '#10b981';
        camCtx.font = 'bold 14px sans-serif';
        camCtx.fillText(`⮂ ${tw.name || tw.line_id}`, mx + 8, my - 8);
      });

      // 5. Черчение текущей зоны или линии
      if (currentPoints.length > 0) {
        if (currentDrawingType === 'tripwire') {
          const p0 = currentPoints[0];
          camCtx.save();
          camCtx.setLineDash([6, 4]);
          camCtx.strokeStyle = '#10b981';
          camCtx.lineWidth = 3;
          camCtx.beginPath();
          camCtx.moveTo(p0[0], p0[1]);
          camCtx.lineTo(mousePos.x, mousePos.y);
          camCtx.stroke();
          camCtx.restore();

          camCtx.fillStyle = '#10b981';
          camCtx.beginPath(); camCtx.arc(p0[0], p0[1], 6, 0, Math.PI * 2); camCtx.fill();
          camCtx.fillStyle = '#ffffff';
          camCtx.font = 'bold 11px sans-serif';
          camCtx.fillText('1 (Старт)', p0[0] + 8, p0[1] - 4);

          camCtx.fillStyle = '#34d399';
          camCtx.font = 'bold 12px sans-serif';
          camCtx.fillText('Кликните 2-ю точку (Конец линии)', mousePos.x + 12, mousePos.y + 16);
        } else {
          camCtx.strokeStyle = '#f59e0b';
          camCtx.fillStyle = 'rgba(245, 158, 11, 0.2)';
          camCtx.lineWidth = 2;
          camCtx.beginPath();
          currentPoints.forEach((p, idx) => {
            if (idx === 0) camCtx.moveTo(p[0], p[1]); else camCtx.lineTo(p[0], p[1]);
          });
          camCtx.lineTo(mousePos.x, mousePos.y);
          camCtx.stroke();

          currentPoints.forEach((p, idx) => {
            camCtx.fillStyle = '#f59e0b';
            camCtx.beginPath();
            camCtx.arc(p[0], p[1], 5, 0, Math.PI * 2);
            camCtx.fill();
            camCtx.fillStyle = '#ffffff';
            camCtx.font = '11px sans-serif';
            camCtx.fillText(`${idx+1}`, p[0] + 7, p[1] + 4);
          });
        }
      }
    }

    function saveZonesToServer() {
      if ((editorZones.dwell_zones || []).length === 0 && (editorZones.tripwires || []).length === 0 && (editorZones.queue_zones || []).length === 0) {
        if (!confirm('Вы действительно хотите сохранить разметку без зон и линий?')) return;
      }
      fetch('/api/zones/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editorZones)
      })
      .then(r => r.json())
      .then(res => {
        const toast = document.getElementById('toast');
        toast.textContent = res.message || 'Разметка сохранена!';
        toast.style.display = 'block';
        setTimeout(() => { toast.style.display = 'none'; }, 3000);
        fetch('/api/config').then(r => r.json()).then(data => { if (data.floor_width_meters) cfg = data; });
        loadEditorZones();
      })
      .catch(err => {
        alert('Ошибка при сохранении разметки: ' + err);
      });
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(content=DASHBOARD_HTML)
