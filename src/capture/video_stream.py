import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class VideoStream:
    """
    Потокобезопасный модуль захвата кадров (RTSP / USB / Видеофайл).
    Поддерживает принцип Zero-Video: хранит в оперативной памяти только
    текущий обрабатываемый кадр без записи на диск.
    """

    def __init__(
        self,
        source: Union[str, int] = "data/sample_video.mp4",
        target_fps: int = 30,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
        loop_video: bool = False,
        threaded: bool = False,
    ):
        self.source = int(source) if str(source).isdigit() else source
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.loop_video = loop_video
        self.threaded = threaded

        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self.frame_index = 0

    def start(self) -> "VideoStream":
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видеопоток: {self.source}")

        if self.threaded:
            self._stopped = False
            self._thread = threading.Thread(target=self._update, daemon=True)
            self._thread.start()
        return self

    def _update(self) -> None:
        frame_interval = 1.0 / max(1, self.target_fps)
        while not self._stopped:
            start_t = time.monotonic()
            ret, frame = self._read_raw()
            if not ret:
                self._stopped = True
                break
            with self._lock:
                self._frame = frame
                self.frame_index += 1
            elapsed = time.monotonic() - start_t
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            if self.loop_video and isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                return False, None

        if self.width and self.height:
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
        return True, frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Возвращает (success, frame).
        В синхронном режиме (для точного теста по кадрам) читает напрямую из VideoCapture.
        """
        if self.threaded:
            with self._lock:
                if self._frame is None:
                    return False, None
                frame_copy = self._frame.copy()
                self._frame = None  # Сброс ссылки после выдачи
                return True, frame_copy
        else:
            ret, frame = self._read_raw()
            if ret:
                self.frame_index += 1
            return ret, frame

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        with self._lock:
            self._frame = None
