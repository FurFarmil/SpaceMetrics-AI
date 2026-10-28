from typing import List, Sequence, Tuple
import cv2
import numpy as np


class HomographyProjector:
    """
    Модуль проекции координат (Homography).
    Преобразует пиксельные координаты стоп человека [u, v] на кадре камеры
    в физические координаты плана помещения [X_meter, Y_meter] через матрицу H (3x3).
    """

    def __init__(
        self,
        src_pixel_points: Sequence[Sequence[float]],
        dst_meter_points: Sequence[Sequence[float]],
    ):
        src = np.array(src_pixel_points, dtype=np.float32)
        dst = np.array(dst_meter_points, dtype=np.float32)
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError("Для калибровки гомографии необходимо ровно 4 опорные точки (4x2).")

        self.matrix_h, _ = cv2.findHomography(src, dst)
        if self.matrix_h is None:
            raise RuntimeError("Не удалось вычислить матрицу гомографии H по заданным точкам.")

        self.matrix_h_inv, _ = cv2.findHomography(dst, src)

    def pixel_to_meter(self, u: float, v: float) -> Tuple[float, float]:
        """
        Умножение [u, v, 1]^T на матрицу калибровки H (3x3) -> [X_meter, Y_meter].
        """
        pt = np.array([[[u, v]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.matrix_h)
        x_m, y_m = projected[0, 0]
        return float(x_m), float(y_m)

    def batch_pixel_to_meter(
        self, points_uv: Sequence[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not points_uv:
            return []
        pts = np.array([points_uv], dtype=np.float32)
        projected = cv2.perspectiveTransform(pts, self.matrix_h)[0]
        return [(float(p[0]), float(p[1])) for p in projected]

    def meter_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Обратное преобразование из координат плана пола (м) в пиксели кадра (для отладки).
        """
        pt = np.array([[[x_m, y_m]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.matrix_h_inv)
        u, v = projected[0, 0]
        return int(round(u)), int(round(v))
