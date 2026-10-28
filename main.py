import argparse
import json
import os
import threading
import time
from typing import Any, Dict, List, Tuple
import cv2
import uvicorn
import yaml

from src.capture.video_stream import VideoStream
from src.inference.base import BaseDetector
from src.inference.yolo_onnx import YOLODetector
from src.inference.hailo_infer import HailoDetector
from src.tracking.tracker import ObjectTracker
from src.geometry.homography import HomographyProjector
from src.analytics.heatmap import HeatmapAccumulator
from src.analytics.dwell_time import DwellTimeTracker
from src.analytics.counter import TripwireAndQueueAnalytics
from src.api.app import app, metrics_store
from src.utils.demo_generator import ensure_floor_plan, ensure_sample_video
from src.utils.visualizer import StreamVisualizer


def load_configs(
    config_path: str = "configs/config.yaml",
    zones_path: str = "configs/zones.json",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(zones_path, "r", encoding="utf-8") as f:
        zones = json.load(f)
    return cfg, zones


def build_detector(inf_cfg: Dict[str, Any]) -> BaseDetector:
    backend = inf_cfg.get("backend", "yolo").lower()
    if backend == "hailo":
        return HailoDetector(
            hef_path=inf_cfg.get("model_path", "models/yolov8n_hailo10h.hef"),
            conf_threshold=float(inf_cfg.get("conf_threshold", 0.35)),
            person_class_id=int(inf_cfg.get("person_class_id", 0)),
        )
    return YOLODetector(
        model_path=inf_cfg.get("model_path", "yolov8n.pt"),
        conf_threshold=float(inf_cfg.get("conf_threshold", 0.35)),
        iou_threshold=float(inf_cfg.get("iou_threshold", 0.45)),
        person_class_id=int(inf_cfg.get("person_class_id", 0)),
    )


def run_pipeline(
    cfg: Dict[str, Any],
    zones: Dict[str, Any],
    video_override: str | None = None,
    max_frames: int = 120,
    anchor_mode: str = "bottom_center",
    save_validation_artifacts: bool = True,
) -> Dict[str, Any]:
    cap_cfg = cfg["capture"]
    inf_cfg = cfg["inference"]
    trk_cfg = cfg["tracking"]
    geo_cfg = cfg["geometry"]
    ana_cfg = cfg["analytics"]

    video_source = video_override if video_override else cap_cfg["source"]

    # Подготовка демо-плана помещения и тестового видео (если еще нет)
    floor_plan_path = ensure_floor_plan(
        "data/floor_plan.png",
        zones_cfg=zones,
        floor_w=float(geo_cfg["floor_width_meters"]),
        floor_h=float(geo_cfg["floor_height_meters"]),
    )
    if isinstance(video_source, str) and not str(video_source).isdigit():
        ensure_sample_video(video_source)

    print(f"[SpaceMetrics AI] Инициализация детектора ({inf_cfg['backend']} -> {inf_cfg['model_path']})...")
    detector = build_detector(inf_cfg)

    tracker = ObjectTracker(
        track_activation_threshold=float(trk_cfg.get("track_Activation_threshold", 0.25)),
        lost_track_buffer=int(trk_cfg.get("lost_track_buffer", 30)),
        minimum_matching_threshold=float(trk_cfg.get("minimum_matching_threshold", 0.8)),
        frame_rate=int(trk_cfg.get("frame_rate", 30)),
    )

    h_calib = zones["homography_calibration"]
    projector = HomographyProjector(
        src_pixel_points=h_calib["src_pixel_points"],
        dst_meter_points=h_calib["dst_meter_points"],
    )
    metrics_store.set_projector(projector)

    ana_hm = ana_cfg.get("heatmap", {})
    heatmap_acc = HeatmapAccumulator(
        floor_width_meters=float(geo_cfg["floor_width_meters"]),
        floor_height_meters=float(geo_cfg["floor_height_meters"]),
        grid_width=int(ana_hm.get("grid_width", 60)),
        grid_height=int(ana_hm.get("grid_height", 50)),
        footprint_radius_m=float(ana_hm.get("footprint_radius_m", 0.45)),
        decay_half_life_sec=float(ana_hm.get("decay_half_life_sec", 0.0)),
    )

    dwell_tracker = DwellTimeTracker(
        zones_config=zones.get("dwell_zones", []),
        default_threshold_sec=float(ana_cfg["dwell_time"]["default_threshold_sec"]),
    )

    counter_analytics = TripwireAndQueueAnalytics(
        tripwires_config=zones.get("tripwires", []),
        queue_zones_config=zones.get("queue_zones", []),
        default_queue_threshold=int(ana_cfg["queue"]["default_alert_threshold"]),
    )

    metrics_store.set_zones_config(zones, geo_cfg)

    stream = VideoStream(
        source=video_source,
        target_fps=int(cap_cfg.get("target_fps", 30)),
        width=int(cap_cfg.get("width", 1280)),
        height=int(cap_cfg.get("height", 720)),
        loop_video=bool(cap_cfg.get("loop_video", True)),
        threaded=False,
    ).start()

    target_fps = float(cap_cfg.get("target_fps", 30.0))
    dt_sec = 1.0 / max(1.0, target_fps)
    all_events_log: List[Dict[str, Any]] = []
    recent_events: List[Dict[str, Any]] = []
    unique_track_ids = set()

    print(
        f"[SpaceMetrics AI] Старт пайплайна | Видео: {video_source} | Режим привязки: {anchor_mode} | "
        f"Лимит кадров: {max_frames if max_frames > 0 else 'бесконечно'}"
    )

    t_start = time.perf_counter()
    frames_processed = 0
    visualizer = StreamVisualizer(max_trail_len=25)

    try:
        while True:
            if 0 < max_frames <= frames_processed:
                break

            t_frame = time.perf_counter()
            ret, frame = stream.read()
            if not ret or frame is None:
                break

            frames_processed += 1

            # 1. Детекция людей (YOLOv8n / HailoRT)
            raw_detections = detector.infer(frame)

            # 2. Трекинг объектов (ByteTrack)
            tracked_objects = tracker.update(raw_detections)

            # 4. Проекция координат стоп/центра (Homography: [u, v] -> [X_meter, Y_meter])
            active_tracks_meters: List[Tuple[int, Tuple[float, float]]] = []
            floor_tracks_payload: List[Dict[str, Any]] = []

            for det in tracked_objects:
                if det.track_id is None:
                    continue
                unique_track_ids.add(det.track_id)
                u, v = det.get_anchor_point(mode=anchor_mode)
                x_m, y_m = projector.pixel_to_meter(u, v)
                # Ограничиваем в пределах зала для корректной аналитики
                x_m = float(max(0.0, min(geo_cfg["floor_width_meters"] - 0.01, x_m)))
                y_m = float(max(0.0, min(geo_cfg["floor_height_meters"] - 0.01, y_m)))

                active_tracks_meters.append((det.track_id, (x_m, y_m)))
                floor_tracks_payload.append(
                    {
                        "track_id": det.track_id,
                        "x_m": round(x_m, 2),
                        "y_m": round(y_m, 2),
                        "conf": round(det.conf, 2),
                    }
                )

            # Проверка динамического обновления зон из визуального редактора
            if metrics_store.check_and_clear_zones_updated():
                print("[SpaceMetrics AI] 🔄 Применена новая разметка зон из веб-редактора!")
                _, zones = load_configs("configs/config.yaml", "configs/zones.json")
                h_calib = zones["homography_calibration"]
                projector = HomographyProjector(
                    src_pixel_points=h_calib["src_pixel_points"],
                    dst_meter_points=h_calib["dst_meter_points"],
                )
                metrics_store.set_projector(projector)
                dwell_tracker = DwellTimeTracker(
                    zones_config=zones.get("dwell_zones", []),
                    default_threshold_sec=float(ana_cfg["dwell_time"]["default_threshold_sec"]),
                )
                counter_analytics = TripwireAndQueueAnalytics(
                    tripwires_config=zones.get("tripwires", []),
                    queue_zones_config=zones.get("queue_zones", []),
                    default_queue_threshold=int(ana_cfg["queue"]["default_alert_threshold"]),
                )
                metrics_store.set_zones_config(zones, geo_cfg)

            # 5. Аналитический движок
            now_ts = time.time()
            # Обновление аккумулятора тепловых карт (Гауссов сплэтинг + Dwell/Flow)
            heatmap_acc.update(active_tracks_meters, dt_seconds=dt_sec)

            dwell_events, dwell_summary = dwell_tracker.update(
                active_tracks_meters, dt_seconds=dt_sec, current_timestamp=now_ts
            )
            tw_q_events, tripwire_stats, queue_stats = counter_analytics.update(
                active_tracks_meters, current_timestamp=now_ts
            )

            frame_events = dwell_events + tw_q_events
            if frame_events:
                all_events_log.extend(frame_events)
                recent_events = (recent_events + frame_events)[-15:]

            elapsed_frame = max(1e-6, time.perf_counter() - t_frame)
            inst_fps = 1.0 / elapsed_frame

            # 6. Экспорт обезличенных метаданных (JSON ~1-2 КБ)
            telemetry_payload: Dict[str, Any] = {
                "status": "running",
                "timestamp": int(now_ts),
                "frame_index": frames_processed,
                "fps": round(inst_fps, 1),
                "active_people_count": len(active_tracks_meters),
                "total_unique_visitors": len(unique_track_ids),
                "total_dwell_events": dwell_tracker.total_dwell_events,
                "floor_tracks": floor_tracks_payload,
                "dwell_zones": dwell_summary,
                "tripwires": tripwire_stats,
                "queues": queue_stats,
                "recent_events": recent_events[-5:],
            }

            if frames_processed % 5 == 0 or frames_processed == 1:
                heatmaps = {
                    "total": heatmap_acc.get_normalized_matrix("total"),
                    "dwell": heatmap_acc.get_normalized_matrix("dwell"),
                    "flow": heatmap_acc.get_normalized_matrix("flow"),
                }
                zone_attention = heatmap_acc.compute_zone_attention(
                    zones.get("dwell_zones", []), mode="dwell"
                )
                metrics_store.update(
                    telemetry_payload,
                    heatmaps,
                    zone_attention=zone_attention,
                )

            # 7. Генерация аннотированного видеокадра Live View для веб-интерфейса
            overlay_opts = metrics_store.get_overlay_config()
            annotated_frame = visualizer.annotate(
                frame=frame,
                tracked_objects=tracked_objects,
                frame_idx=frames_processed,
                fps=inst_fps,
                projector=projector,
                zones_config=zones,
                telemetry=telemetry_payload,
                overlay_options=overlay_opts,
            )
            _, enc_live = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            metrics_store.set_annotated_frame(enc_live.tobytes())

            # Сохраняем чистый стоп-кадр для редактора зон (периодически)
            if frames_processed % 10 == 0 or frames_processed == 1:
                _, enc_clean = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                metrics_store.set_camera_frame(enc_clean.tobytes(), frame.shape[1], frame.shape[0])

            # Архитектурный принцип Zero-Video: немедленное удаление сырого и аннотированного кадров из RAM
            del frame, annotated_frame

            if frames_processed % 30 == 0 or frames_processed == 1:
                print(
                    f"  Кадр #{frames_processed:04d} | FPS: {inst_fps:5.1f} | "
                    f"Людей в кадре: {len(active_tracks_meters):2d} | "
                    f"Уникальных треков: {len(unique_track_ids):2d} | "
                    f"Событий: {len(all_events_log)}"
                )

    finally:
        stream.stop()

    total_time = max(1e-6, time.perf_counter() - t_start)
    avg_fps = frames_processed / total_time

    zone_attention_final = heatmap_acc.compute_zone_attention(
        zones.get("dwell_zones", []), mode="dwell"
    )

    summary_report = {
        "project": "SpaceMetrics AI MVP",
        "privacy_mode": "Zero-Video (Raw frames dropped immediately)",
        "frames_processed": frames_processed,
        "average_fps": round(avg_fps, 2),
        "total_unique_tracks": len(unique_track_ids),
        "total_dwell_events": dwell_tracker.total_dwell_events,
        "zone_attention_breakdown": zone_attention_final,
        "tripwires_final": metrics_store.get_snapshot().get("tripwires", {}),
        "queues_final": metrics_store.get_snapshot().get("queues", {}),
        "dwell_zones_final": metrics_store.get_snapshot().get("dwell_zones", {}),
        "sample_events": all_events_log[:25],
    }

    if save_validation_artifacts:
        os.makedirs("output", exist_ok=True)
        floor_img = cv2.imread(floor_plan_path)

        # 1. Total Cumulative Heatmap (Общая активность на полу)
        hm_total = heatmap_acc.render_on_floor_plan(
            floor_img,
            mode="total",
            colormap=cv2.COLORMAP_TURBO,
            zones_config=zones.get("dwell_zones", []),
            title="TOTAL PRESENCE HEATMAP (TURBO)",
        )
        total_out_path = os.path.join("output", "heatmap_total.png")
        cv2.imwrite(total_out_path, hm_total)
        cv2.imwrite(os.path.join("output", "heatmap_validation.png"), hm_total)

        # 2. Dwell Heatmap (Зоны внимания и остановки у витрин)
        hm_dwell = heatmap_acc.render_on_floor_plan(
            floor_img,
            mode="dwell",
            colormap=cv2.COLORMAP_JET,
            zones_config=zones.get("dwell_zones", []),
            title="DWELL & ATTENTION HEATMAP (STOP / SHOPPING)",
        )
        dwell_out_path = os.path.join("output", "heatmap_dwell.png")
        cv2.imwrite(dwell_out_path, hm_dwell)

        # 3. Flow Heatmap (Транзитные коридоры движения)
        hm_flow = heatmap_acc.render_on_floor_plan(
            floor_img,
            mode="flow",
            colormap=cv2.COLORMAP_INFERNO,
            zones_config=zones.get("dwell_zones", []),
            title="TRANSIT FLOW & FOOTPATH HEATMAP",
        )
        flow_out_path = os.path.join("output", "heatmap_flow.png")
        cv2.imwrite(flow_out_path, hm_flow)

        json_out_path = os.path.join("output", "metrics_log.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)

        print(f"\n[Валидация завершена]")
        print(f"  -> Обработано кадров: {frames_processed} (средний FPS: {avg_fps:.1f})")
        print(f"  -> Уникальных посетителей (track_id): {len(unique_track_ids)}")
        print(f"  -> [OK] 2D Heatmap Total:    {total_out_path}")
        print(f"  -> [OK] 2D Heatmap Dwell:    {dwell_out_path}")
        print(f"  -> [OK] 2D Heatmap Flow:     {flow_out_path}")
        print(f"  -> [OK] JSON Метрики & Доли: {json_out_path}")

    return summary_report


def main() -> None:
    parser = argparse.ArgumentParser(description="SpaceMetrics AI — MVP Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Путь к config.yaml")
    parser.add_argument("--zones", type=str, default="configs/zones.json", help="Путь к zones.json")
    parser.add_argument("--video", type=str, default=None, help="Путь к видеофайлу или RTSP/USB источнику")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Количество кадров для прогона (0 = без лимита)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        choices=["bottom_center", "center"],
        default="bottom_center",
        help="Точка привязки: bottom_center (стопы, камера под углом) или center (строго сверху)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Запустить FastAPI + WebSocket сервер и веб-дашборд параллельно с обработкой",
    )
    args = parser.parse_args()

    cfg, zones = load_configs(args.config, args.zones)

    if args.serve:
        api_cfg = cfg.get("api", {})
        host = api_cfg.get("host", "127.0.0.1")
        port = int(api_cfg.get("port", 8000))
        print(f"[FastAPI] Запуск дашборда и WebSocket-сервера на http://{host}:{port} ...")
        server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": app, "host": host, "port": port, "log_level": "warning"},
            daemon=True,
        )
        server_thread.start()

    run_pipeline(
        cfg=cfg,
        zones=zones,
        video_override=args.video,
        max_frames=args.max_frames,
        anchor_mode=args.anchor,
        save_validation_artifacts=True,
    )


if __name__ == "__main__":
    main()
