import os
import urllib.request
from typing import Dict, Any
import cv2
import numpy as np


SAMPLE_VIDEO_URLS = [
    # Официальный тестовый ролик пешеходов из репозитория OpenCV (768x576, люди идут по залу/переходу)
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
]


def ensure_floor_plan(
    output_path: str = "data/floor_plan.png",
    zones_cfg: Dict[str, Any] | None = None,
    floor_w: float = 12.0,
    floor_h: float = 10.0,
) -> str:
    """
    Создает наглядную схему помещения (2D Floor Plan) с размеченными зонами
    витрин, касс и линий входа, если файл еще не существует.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w_px, h_px = 720, 600
    img = np.full((h_px, w_px, 3), 248, dtype=np.uint8)

    def m2px(xm: float, ym: float) -> tuple[int, int]:
        return int(xm / floor_w * w_px), int(ym / floor_h * h_px)

    # Координатная сетка 1х1 метр
    for xm in range(0, int(floor_w) + 1):
        x, _ = m2px(xm, 0)
        cv2.line(img, (x, 0), (x, h_px), (225, 225, 225), 1)
    for ym in range(0, int(floor_h) + 1):
        _, y = m2px(0, ym)
        cv2.line(img, (0, y), (w_px, y), (225, 225, 225), 1)

    # Рамка зала
    cv2.rectangle(img, (2, 2), (w_px - 3, h_px - 3), (80, 90, 110), 2)

    if zones_cfg:
        # Отрисовка Dwell зон (витрины)
        for dz in zones_cfg.get("dwell_zones", []):
            pts = np.array([m2px(p[0], p[1]) for p in dz["polygon"]], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (255, 235, 205))
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.polylines(img, [pts], True, (210, 120, 20), 2)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            cv2.putText(
                img,
                dz["zone_id"],
                (max(10, cx - 60), cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (140, 70, 10),
                1,
                cv2.LINE_AA,
            )

        # Отрисовка Queue зон (кассы)
        for qz in zones_cfg.get("queue_zones", []):
            pts = np.array([m2px(p[0], p[1]) for p in qz["polygon"]], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (220, 230, 255))
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.polylines(img, [pts], True, (50, 80, 200), 2)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            cv2.putText(
                img,
                qz["zone_id"],
                (max(10, cx - 60), cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 50, 150),
                1,
                cv2.LINE_AA,
            )

        # Отрисовка Tripwire линий
        for tw in zones_cfg.get("tripwires", []):
            p1 = m2px(tw["start"][0], tw["start"][1])
            p2 = m2px(tw["end"][0], tw["end"][1])
            cv2.line(img, p1, p2, (40, 180, 90), 2, cv2.LINE_AA)
            cv2.putText(
                img,
                f"Tripwire: {tw['line_id']}",
                (p1[0] + 10, p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (25, 130, 60),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, img)
    return output_path


def ensure_sample_video(output_path: str = "data/sample_video.mp4") -> str:
    """
    Проверяет наличие тестового видео.
    Если видео отсутствует — скачивает реальный тестовый ролик с пешеходами (OpenCV vtest.avi)
    и перекодирует/сохраняет в data/sample_video.mp4.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_avi = output_path + ".temp.avi"

    for url in SAMPLE_VIDEO_URLS:
        try:
            print(f"[Demo Setup] Скачивание тестового видео с людьми: {url} ...")
            urllib.request.urlretrieve(url, temp_avi)
            if os.path.exists(temp_avi) and os.path.getsize(temp_avi) > 10_000:
                cap = cv2.VideoCapture(temp_avi)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))
                frames_written = 0
                # Берем первые 250 кадров для быстрого тестового прогона
                while frames_written < 250:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized = cv2.resize(frame, (1280, 720))
                    writer.write(resized)
                    frames_written += 1
                cap.release()
                writer.release()
                os.remove(temp_avi)
                if frames_written > 0:
                    print(f"[Demo Setup] Готово: {output_path} ({frames_written} кадров, 1280x720)")
                    return output_path
        except Exception as exc:
            print(f"[Demo Setup] Не удалось скачать с {url}: {exc}")
            if os.path.exists(temp_avi):
                os.remove(temp_avi)

    raise RuntimeError(
        f"Не удалось автоматически подготовить {output_path}. Пожалуйста, положите любой .mp4 файл в data/sample_video.mp4."
    )
