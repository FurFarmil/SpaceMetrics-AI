import os
import sys
import urllib.request
import cv2


BENCHMARK_SOURCES = {
    "retail_store_aisle.mp4": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4",
        "description": "Съемка прохода в магазине (Retail Store Aisle) — мониторинг витрин, полок и движения покупателей",
        "type": "retail_aisle",
    },
    "pedestrians_mall_passage.mp4": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        "description": "Пешеходный проход / торговая галерея со стационарной камеры (потоки людей, плотность, пересечения)",
        "type": "crowd_passage",
    },
    "store_walk_and_dwell.mp4": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4",
        "description": "Покупатели идут и останавливаются перед витриной (Dwell Time & Interest Tracking)",
        "type": "dwell_pause",
    },
    "entrance_turnstile_count.mp4": {
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4",
        "description": "Проход людей по одному через входную зону (Tripwire In/Out Counting)",
        "type": "entrance_counting",
    },
    "pedestrians_cctv_long.mp4": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
        "description": "Полная длительная запись CCTV видеонаблюдения за пешеходами (795 кадров) для стресс-тестирования",
        "type": "cctv_long",
    },
}


def download_progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        percent = min(100.0, float(block_num * block_size) / total_size * 100.0)
        sys.stdout.write(f"\r  Прогресс: {percent:5.1f}% [{block_num * block_size // 1024} KB / {total_size // 1024} KB]")
        sys.stdout.flush()


def download_benchmarks(target_dir: str = "data/benchmarks") -> None:
    os.makedirs(target_dir, exist_ok=True)
    print(f"=== Загрузка бенчмарк-видео для SpaceMetrics AI в {target_dir} ===\n")

    for filename, info in BENCHMARK_SOURCES.items():
        out_path = os.path.join(target_dir, filename)
        url = info["url"]
        desc = info["description"]

        if os.path.exists(out_path) and os.path.getsize(out_path) > 50_000:
            cap = cv2.VideoCapture(out_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"[УЖЕ СКАЧАН] {filename} ({w}x{h}, {total_frames} кадров, {fps:.1f} FPS)\n  -> {desc}\n")
            continue

        print(f"[СКАЧИВАНИЕ] {filename}...")
        print(f"  Назначение: {desc}")
        print(f"  URL: {url}")

        temp_path = out_path + ".downloading"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f_out:
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 64 * 1024
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = min(100.0, downloaded / total_size * 100.0)
                        sys.stdout.write(f"\r  Загружено: {pct:5.1f}% ({downloaded // 1024} KB / {total_size // 1024} KB)")
                        sys.stdout.flush()

            sys.stdout.write("\n")

            # Если файл AVI, конвертируем в чистый mp4 с помощью OpenCV
            if filename.endswith(".mp4") and url.endswith(".avi"):
                cap = cv2.VideoCapture(temp_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                frames_cnt = 0
                while True:
                    ret, fr = cap.read()
                    if not ret:
                        break
                    writer.write(fr)
                    frames_cnt += 1
                cap.release()
                writer.release()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                if os.path.exists(out_path):
                    os.remove(out_path)
                os.rename(temp_path, out_path)

            cap = cv2.VideoCapture(out_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"  [OK] Сохранен: {out_path} ({w}x{h}, {total_frames} кадров, {fps:.1f} FPS)\n")

        except Exception as exc:
            print(f"  [ОШИБКА] Не удалось скачать {url}: {exc}\n")
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/benchmarks"
    download_benchmarks(target)
