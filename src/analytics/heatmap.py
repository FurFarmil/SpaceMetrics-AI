from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class HeatmapAccumulator:
    """
    Профессиональный аналитический аккумулятор 2D-тепловой карты для ритейла (SpaceMetrics AI).
    
    Ключевые возможности:
    1. Physical Gaussian Footprint Splatting: каждый человек оставляет физический след
       заданного радиуса (по умолчанию 0.45м), а не единичный пиксель.
    2. Скоростная дифференциация (Dwell vs Traffic):
       - Dwell Heatmap: подчеркивает зоны остановки и внимания покупателей (инверсно к скорости).
       - Traffic/Flow Heatmap: подчеркивает основные транзитные коридоры и маршруты движения.
       - Total Heatmap: суммарное время присутствия.
    3. Траекторная интерполяция: устраняет прерывистость при быстром движении или пропуске кадров.
    4. Временное затухание (Temporal Half-life Decay): экспоненциальное остывание старых следов.
    5. Zone Attention Share: расчет доли внимания (в %) по витринам и зонам зала.
    """

    def __init__(
        self,
        floor_width_meters: float = 12.0,
        floor_height_meters: float = 10.0,
        grid_width: int = 60,
        grid_height: int = 50,
        footprint_radius_m: float = 0.45,
        decay_half_life_sec: float = 0.0,
    ):
        self.floor_width = float(floor_width_meters)
        self.floor_height = float(floor_height_meters)
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.footprint_radius_m = float(footprint_radius_m)
        self.decay_half_life_sec = float(decay_half_life_sec)

        self.cell_size_x = self.floor_width / self.grid_width
        self.cell_size_y = self.floor_height / self.grid_height

        # Матрицы накопления (H_grid, W_grid)
        self.grid_total = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.grid_dwell = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.grid_flow = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # История позиций треков: {track_id: (x_m, y_m)}
        self._prev_track_positions: Dict[int, Tuple[float, float]] = {}

        # Предварительно сгенерированное гауссово ядро следа ноги
        self._kernel, self._k_rad_y, self._k_rad_x = self._build_footprint_kernel()

    def _build_footprint_kernel(self) -> Tuple[np.ndarray, int, int]:
        rad_x = max(1, int(np.ceil(self.footprint_radius_m / self.cell_size_x)))
        rad_y = max(1, int(np.ceil(self.footprint_radius_m / self.cell_size_y)))

        y_coords = np.arange(-rad_y, rad_y + 1) * self.cell_size_y
        x_coords = np.arange(-rad_x, rad_x + 1) * self.cell_size_x
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Двумерный Гаусс с дисперсией sigma = radius / 2.0
        sigma = self.footprint_radius_m / 2.0
        dist_sq = xx**2 + yy**2
        kernel = np.exp(-dist_sq / (2.0 * sigma**2))
        kernel[dist_sq > (self.footprint_radius_m * 1.2) ** 2] = 0.0

        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
        return kernel.astype(np.float32), rad_y, rad_x

    def _splat_point(
        self,
        target_grid: np.ndarray,
        x_m: float,
        y_m: float,
        weight: float,
    ) -> None:
        """Наложение гауссова отпечатка ноги в координаты пола."""
        cx = int(round((x_m / self.floor_width) * (self.grid_width - 1)))
        cy = int(round((y_m / self.floor_height) * (self.grid_height - 1)))

        if not (0 <= cx < self.grid_width and 0 <= cy < self.grid_height):
            return

        # Координаты нарезки для сетки
        y1_grid = max(0, cy - self._k_rad_y)
        y2_grid = min(self.grid_height, cy + self._k_rad_y + 1)
        x1_grid = max(0, cx - self._k_rad_x)
        x2_grid = min(self.grid_width, cx + self._k_rad_x + 1)

        # Соответствующие координаты в ядре
        y1_k = y1_grid - (cy - self._k_rad_y)
        y2_k = y1_k + (y2_grid - y1_grid)
        x1_k = x1_grid - (cx - self._k_rad_x)
        x2_k = x1_k + (x2_grid - x1_grid)

        target_grid[y1_grid:y2_grid, x1_grid:x2_grid] += (
            self._kernel[y1_k:y2_k, x1_k:x2_k] * weight
        )

    def apply_decay(self, dt_seconds: float) -> None:
        """Экспоненциальное затухание старого тепла по периоду полураспада."""
        if self.decay_half_life_sec > 0:
            decay_factor = float(np.exp(-dt_seconds * np.log(2.0) / self.decay_half_life_sec))
            self.grid_total *= decay_factor
            self.grid_dwell *= decay_factor
            self.grid_flow *= decay_factor

    def update(
        self,
        active_tracks: Sequence[Tuple[int, Tuple[float, float]]],
        dt_seconds: float = 1.0 / 30.0,
    ) -> None:
        """
        Обновляет тепловую карту на основе активных треков:
        - Рассчитывает мгновенную скорость каждого посетителя.
        - Интерполирует траекторию между кадрами (плавный непрерывный след).
        - Разделяет вес на Dwell (остановка/внимание) и Flow (транзит).
        """
        self.apply_decay(dt_seconds)

        current_track_ids = {tid for tid, _ in active_tracks}

        for tid, (x_m, y_m) in active_tracks:
            # Ограничение физических координат залом
            x_m = float(np.clip(x_m, 0.0, self.floor_width - 0.01))
            y_m = float(np.clip(y_m, 0.0, self.floor_height - 0.01))

            prev_pos = self._prev_track_positions.get(tid)
            if prev_pos is not None:
                dx = x_m - prev_pos[0]
                dy = y_m - prev_pos[1]
                distance = float(np.hypot(dx, dy))
                speed = distance / max(1e-4, dt_seconds)  # м/с
            else:
                distance = 0.0
                speed = 0.0
                prev_pos = (x_m, y_m)

            # Весовые коэффициенты на базе скорости:
            # - Стоящий/медленный человек (v < 0.3 м/с) дает максимальный вес dwell.
            # - Быстро идущий человек (v > 1.0 м/с) дает вес в flow.
            dwell_weight = float(1.0 / (1.0 + 3.5 * speed))
            flow_weight = float(np.clip(speed / 0.8, 0.1, 1.5))

            # Траекторная интерполяция: если человек переместился больше чем на пол-ячейки
            min_step = min(self.cell_size_x, self.cell_size_y) * 0.7
            if distance > min_step:
                num_substeps = int(np.clip(np.ceil(distance / min_step), 2, 8))
                sub_dt = dt_seconds / num_substeps
                for s in range(1, num_substeps + 1):
                    alpha = s / float(num_substeps)
                    sx = prev_pos[0] + dx * alpha
                    sy = prev_pos[1] + dy * alpha
                    self._splat_point(self.grid_total, sx, sy, sub_dt)
                    self._splat_point(self.grid_dwell, sx, sy, sub_dt * dwell_weight)
                    self._splat_point(self.grid_flow, sx, sy, sub_dt * flow_weight)
            else:
                self._splat_point(self.grid_total, x_m, y_m, dt_seconds)
                self._splat_point(self.grid_dwell, x_m, y_m, dt_seconds * dwell_weight)
                self._splat_point(self.grid_flow, x_m, y_m, dt_seconds * flow_weight)

            self._prev_track_positions[tid] = (x_m, y_m)

        # Очистка неактивных треков
        stale_ids = [tid for tid in self._prev_track_positions if tid not in current_track_ids]
        for tid in stale_ids:
            del self._prev_track_positions[tid]

    def get_raw_grid(self, mode: str = "total") -> np.ndarray:
        if mode == "dwell":
            return self.grid_dwell
        elif mode == "flow":
            return self.grid_flow
        return self.grid_total

    def get_normalized_matrix(
        self,
        mode: str = "total",
        smooth_sigma: float = 0.6,
    ) -> List[List[float]]:
        """
        Возвращает нормализованную матрицу [0.0 .. 1.0] для выбранного режима
        (total / dwell / flow) в виде 2D-списка для передачи по JSON/WebSocket.
        """
        grid = self.get_raw_grid(mode).copy()
        if smooth_sigma > 0 and np.max(grid) > 0:
            grid = gaussian_filter(grid, sigma=smooth_sigma)
        max_val = float(np.max(grid))
        if max_val > 1e-6:
            grid = grid / max_val
        return np.round(grid, 4).tolist()

    def compute_zone_attention(
        self,
        zones_config: Sequence[Dict[str, Any]],
        mode: str = "dwell",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Рассчитывает долю внимания (Attention Share / Heat Energy) для каждого полигона зоны.
        Позволяет ритейлу точно измерить в % распределение интереса посетителей к витринам.
        """
        grid = self.get_raw_grid(mode)
        total_energy = float(np.sum(grid))

        results: Dict[str, Dict[str, Any]] = {}
        # Сетка координат ячеек в метрах
        xs = (np.arange(self.grid_width) + 0.5) * self.cell_size_x
        ys = (np.arange(self.grid_height) + 0.5) * self.cell_size_y
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

        for z in zones_config:
            z_id = z["zone_id"]
            name = z.get("name", z_id)
            poly = np.array(z["polygon"], dtype=np.float32)

            # Проверка принадлежности центров ячеек полигону
            mask = np.zeros(len(pts), dtype=bool)
            for i, p in enumerate(pts):
                if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0:
                    mask[i] = True

            mask_2d = mask.reshape((self.grid_height, self.grid_width))
            zone_energy = float(np.sum(grid[mask_2d]))
            share_pct = (zone_energy / total_energy * 100.0) if total_energy > 1e-6 else 0.0

            results[z_id] = {
                "zone_id": z_id,
                "name": name,
                "accumulated_seconds": round(zone_energy, 2),
                "attention_share_percent": round(share_pct, 1),
            }

        return results

    def render_on_floor_plan(
        self,
        floor_plan_img: Optional[np.ndarray] = None,
        output_size: Tuple[int, int] = (720, 600),
        mode: str = "total",
        colormap: int = cv2.COLORMAP_TURBO,
        zones_config: Optional[Sequence[Dict[str, Any]]] = None,
        title: Optional[str] = None,
    ) -> np.ndarray:
        """
        Генерирует высококачественную графическую визуализацию тепловой карты
        поверх плана зала с настраиваемой палитрой (TURBO, JET, INFERNO) и инфо-панелью.
        """
        w_out, h_out = output_size
        if floor_plan_img is not None:
            base_img = cv2.resize(floor_plan_img, (w_out, h_out))
        else:
            base_img = np.full((h_out, w_out, 3), 245, dtype=np.uint8)
            # Координатная сетка пола 1х1 метр
            for xm in range(int(self.floor_width) + 1):
                px = int(xm / self.floor_width * w_out)
                cv2.line(base_img, (px, 0), (px, h_out), (225, 225, 225), 1)
            for ym in range(int(self.floor_height) + 1):
                py = int(ym / self.floor_height * h_out)
                cv2.line(base_img, (0, py), (w_out, py), (225, 225, 225), 1)

        norm_mat = np.array(self.get_normalized_matrix(mode=mode, smooth_sigma=0.8), dtype=np.float32)
        max_accumulated = float(np.max(self.get_raw_grid(mode)))

        if np.max(norm_mat) > 1e-6:
            u8_mat = np.uint8(np.clip(norm_mat * 255.0, 0, 255))
            u8_resized = cv2.resize(u8_mat, (w_out, h_out), interpolation=cv2.INTER_CUBIC)
            colored = cv2.applyColorMap(u8_resized, colormap)

            # Нелинейный альфа-канал для мягкого градиента (горячие зоны непрозрачнее)
            alpha_map = cv2.resize(norm_mat, (w_out, h_out), interpolation=cv2.INTER_CUBIC)
            alpha_map = np.clip(alpha_map, 0.0, 1.0)
            alpha_map = np.clip(np.power(alpha_map, 0.7) * 0.75, 0.0, 0.8)[..., None]

            result = (base_img.astype(np.float32) * (1.0 - alpha_map) + colored.astype(np.float32) * alpha_map).astype(
                np.uint8
            )
        else:
            result = base_img.copy()

        # Отрисовка зон интереса поверх карты
        if zones_config:
            def m2px(xm: float, ym: float) -> Tuple[int, int]:
                return int(xm / self.floor_width * w_out), int(ym / self.floor_height * h_out)

            for z in zones_config:
                pts = np.array([m2px(p[0], p[1]) for p in z["polygon"]], dtype=np.int32)
                cv2.polylines(result, [pts], True, (40, 40, 220), 2, cv2.LINE_AA)

        # Отрисовка Colorbar (шкалы интенсивности) внизу
        cbar_h, cbar_w = 12, 180
        cbar_x = w_out - cbar_w - 20
        cbar_y = h_out - cbar_h - 15

        gradient = np.linspace(0, 255, cbar_w, dtype=np.uint8).reshape((1, cbar_w))
        gradient_colored = cv2.applyColorMap(gradient, colormap)
        gradient_colored = cv2.resize(gradient_colored, (cbar_w, cbar_h))
        result[cbar_y : cbar_y + cbar_h, cbar_x : cbar_x + cbar_w] = gradient_colored
        cv2.rectangle(result, (cbar_x, cbar_y), (cbar_x + cbar_w, cbar_y + cbar_h), (50, 50, 50), 1)

        cv2.putText(
            result, "Low", (cbar_x - 30, cbar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1
        )
        max_label = f"{max_accumulated:.1f}s" if max_accumulated < 60 else f"{max_accumulated/60:.1f}m"
        cv2.putText(
            result, max_label, (cbar_x + cbar_w + 6, cbar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1
        )

        # Заголовок режима карты
        mode_titles = {
            "total": "CUMULATIVE PRESENCE HEATMAP",
            "dwell": "DWELL & ATTENTION HEATMAP (STOP / SHOPPING)",
            "flow": "TRANSIT FLOW & FOOTPATH HEATMAP",
        }
        header_text = title or mode_titles.get(mode, "HEATMAP")
        cv2.putText(
            result,
            header_text,
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

        return result

    def reset(self) -> None:
        self.grid_total.fill(0.0)
        self.grid_dwell.fill(0.0)
        self.grid_flow.fill(0.0)
        self._prev_track_positions.clear()
