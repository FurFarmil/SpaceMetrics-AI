<div align="center">

# 🏬 SpaceMetrics AI — Edge Retail Spatial Intelligence

### Real-Time Zero-Video Retail Analytics Sensor Powered by ASUS UGen300 & Hailo-10H

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-ASUS%20UGen300%20(Hailo--10H)-00e5ff.svg)](https://www.asus.com/)
[![Zero-Video](https://img.shields.io/badge/Privacy-Zero--Video%20GDPR%20Compliant-10b981.svg)](#-zero-video--privacy-by-design-architecture)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%20%2B%20WebSocket-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An autonomous, privacy-preserving edge sensor that transforms raw retail surveillance video into actionable physical telemetry: multi-modal Gaussian splatting heatmaps, customer dwell & attention share, entrance/exit tripwire counts, queue density alerts, and interactive in-browser zone markup.*

---

**Competition Track:** ASUS UGen AI League — Battlefield Lightning (Lightweight Computing)  
**Target Accelerator:** ASUS UGen300 (Hailo-10H NPU, 40 TOPS INT4, 8 GB LPDDR4X)  
**Host Platform:** ASUS Mini PC / Linux SBC (Ubuntu 22.04 LTS / Debian)

</div>

---

## 📑 Table of Contents

- [Key Highlights](#-key-highlights)
- [Zero-Video & Privacy-by-Design Architecture](#-zero-video--privacy-by-design-architecture)
- [System Architecture & Data Pipeline](#-system-architecture--data-pipeline)
- [Core Capabilities](#-core-capabilities)
  - [1. Physical Multi-Layer Heatmap Engine](#1-physical-multi-layer-heatmap-engine-gaussian-splatting)
  - [2. Interactive In-Browser Camera Zone Editor](#2-interactive-in-browser-camera-zone-editor)
  - [3. Real-Time Live View Tracking Stream](#3-real-time-live-view-tracking-stream)
  - [4. Directional Tripwire Traffic Counter](#4-directional-tripwire-traffic-counter)
  - [5. Dwell Time & Showcase Attention Share](#5-dwell-time--showcase-attention-share)
  - [6. Queue Density & Cashier Alerting](#6-queue-density--cashier-alerting)
- [Hardware & NPU Acceleration (Hailo-10H)](#-hardware--npu-acceleration-hailo-10h)
- [Quick Start Guide](#-quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Edge Pipeline](#running-the-edge-pipeline)
  - [Benchmark Datasets](#benchmark-datasets)
- [Web Dashboard & UI Guide](#-web-dashboard--ui-guide)
- [REST API & WebSocket Documentation](#-rest-api--websocket-documentation)
- [Project Structure](#-project-structure)
- [Configuration Reference](#-configuration-reference)
- [License](#-license)

---

## 🌟 Key Highlights

- 🔒 **Zero-Video Architecture (GDPR / Privacy-by-Design):** Raw video frames reside strictly in volatile RAM for milliseconds and are immediately destroyed after inference. Only anonymized JSON events (1–2 KB/s) and 2D heatmap matrices leave the sensor.
- 📐 **Dual-Coordinate Persistence:** Visual zones drawn on the camera canvas preserve exact pixel coordinates ($1280 \times 720$) for pixel-perfect Live View streaming, while simultaneously mapping to real-world floor meters ($12\text{m} \times 10\text{m}$) via Homography for analytics.
- 🔥 **Physical Gaussian Footprint Splatting:** Continuous foot-anchor trajectory interpolation with configurable radius ($0.45\text{m}$) and half-life decay across three distinct layers: **Total Presence**, **Dwell/Attention**, and **Transit Flow**.
- ✏️ **Interactive In-Browser Zone Editor:** Draw custom dwell showcases, cashier queue zones, and entrance/exit tripwires directly over the live camera view using intuitive mouse clicks — instantly applied to the running pipeline without restarts.
- 🎥 **Low-Latency Live View MJPEG Stream:** High-speed streaming (~25–30 FPS) with toggleable OpenCV visual overlays: bounding boxes, trajectory breadcrumbs, ground-contact foot anchors with physical meter coordinates, and real-time HUD telemetry.
- ⚡ **Tailored for ASUS UGen300:** Engineered for the Hailo-10H M.2 / PCIe accelerator with native `hailort` PyHailoRT runtime support and seamless fallbacks to ONNX Runtime and PyTorch YOLOv8n.

---

## 🛡️ Zero-Video & Privacy-by-Design Architecture

Traditional retail video analytics solutions transmit heavy video streams (5–20 Mbps per camera) to centralized cloud servers, incurring severe bandwidth costs and massive GDPR/CCPA compliance risks.

**SpaceMetrics AI operates under a strict Zero-Video paradigm:**

```
Raw Camera Frame (1080p/720p)
   │ (In-Memory Processing Only)
   ▼
Object Detection + ByteTrack ──► [Frame Immediately Deleted from RAM]
   │
   ▼
Homography [u, v] ──► [X_meter, Y_meter]
   │
   ▼
Anonymized Telemetry Output Only:
   ├── JSON Event Stream (~1–2 KB/s): IDs, speeds, dwell times, counts
   └── 2D Normalized Heatmap Matrix (60 × 50 float grid)
```

- **No Face Recognition or Biometrics:** Only bounding boxes and foot ground-contact points.
- **Zero Raw Storage:** No MP4 or raw JPEG files are ever stored on disk.
- **Micro-Bandwidth:** Operates over lightweight 4G/LTE/Wi-Fi or constrained industrial edge networks.

---

## 🏗️ System Architecture & Data Pipeline

```mermaid
flowchart TD
    A["Camera Stream (RTSP / USB / MP4)"] --> B["VideoStream Capture (OpenCV / V4L2)"]
    B --> C["Person Detector (Hailo-10H / YOLOv8n)"]
    C --> D["Multi-Object Tracker (ByteTrack)"]
    D --> E["Foot-Anchor Ground Point [u, v]"]
    E --> F["Homography Projector (H: Pixels ➔ Meters)"]
    
    F --> G1["Gaussian Splatting Heatmap Engine"]
    F --> G2["Dwell Time & Attention Tracker"]
    F --> G3["Tripwire Traffic Counter (In/Out)"]
    F --> G4["Queue Density & Cashier Monitor"]
    
    G1 & G2 & G3 & G4 --> H["MetricsStateStore (Thread-Safe Cache)"]
    
    B & D & F --> I["StreamVisualizer (OpenCV Annotator)"]
    I --> H
    
    H --> J1["FastAPI REST Endpoints (/api/metrics, /api/heatmap)"]
    H --> J2["WebSocket Real-Time Telemetry (/ws/metrics)"]
    H --> J3["Live View MJPEG Stream (/api/camera/stream)"]
    H --> J4["Interactive Web Dashboard & Zone Editor"]
```

---

## 💡 Core Capabilities

### 1. Physical Multi-Layer Heatmap Engine (Gaussian Splatting)
Implemented in [`src/analytics/heatmap.py`](file:///c:/ДЗ/ASUS/src/analytics/heatmap.py):
- **Physical 2D Footprint:** Disperses continuous 2D Gaussian density centered at each visitor's physical floor coordinates $[X_m, Y_m]$ with customizable radius ($r=0.45\text{m}$).
- **Trajectory Interpolation:** Sub-samples path segments between frames, preventing discrete "bead-string" artifacts during rapid walking or dropped frames.
- **Three Operational Layers:**
  - `total`: Cumulative store presence (Turbo palette).
  - `dwell`: Customer dwell and showcase browsing (Jet palette; inversely weighted by movement velocity).
  - `flow`: Foot traffic corridors and transit pathways (Inferno palette).
- **Half-Life Decay:** Configurable temporal cooling (`decay_half_life_sec`) for continuous live operations.

### 2. Interactive In-Browser Camera Zone Editor
Located in Tab 3 of the dashboard:
- Draw directly on top of the live camera view:
  - **Showcases (Dwell Zones):** Multi-point polygons with customizable dwell thresholds.
  - **Cashier Lines (Queue Zones):** Multi-point polygons with occupancy thresholds.
  - **Entrance/Exit Lines (Tripwires):** Two-point lines with automatic directional arrows (`IN` vs `OUT`).
- **Hot Reloading:** When clicking **"Сохранить и применить в систему"**, changes are written to [`configs/zones.json`](file:///c:/ДЗ/ASUS/configs/zones.json) and instantly reloaded into the active analytics engine without dropping frames.

### 3. Real-Time Live View Tracking Stream
Located in Tab 2 and Split View mode:
- Low-latency MJPEG stream rendered directly via [`src/utils/visualizer.py`](file:///c:/ДЗ/ASUS/src/utils/visualizer.py).
- **Visual Overlays:**
  - Bounding boxes with tracking IDs and confidence metrics.
  - Smooth multi-frame trajectory breadcrumb trails.
  - Ground-contact foot anchors with live $[X_m, Y_m]$ labels.
  - Synchronized zone polygons and tripwire crossing indicators.
  - Upper Heads-Up Display (HUD) showing instantaneous FPS, active visitor count, total visits, and tripwire counts.
- **Layer Toggle Panel:** Dynamically enable or disable boxes, trails, anchors, zones, or HUD on the fly.

### 4. Directional Tripwire Traffic Counter
Implemented in [`src/analytics/counter.py`](file:///c:/ДЗ/ASUS/src/analytics/counter.py):
- Detects line segment intersection between a visitor's previous and current floor position:
  $$\vec{P}_{\text{prev}} \to \vec{P}_{\text{curr}} \quad \cap \quad \vec{L}_{\text{start}} \to \vec{L}_{\text{end}}$$
- Uses counter-clockwise orientation ($\text{CCW}$) to distinguish between entrance (`IN`) and exit (`OUT`) directions.
- Built-in debouncing ($0.8\text{s}$) prevents duplicate trigger events from lingering footsteps.

### 5. Dwell Time & Showcase Attention Share
Implemented in [`src/analytics/dwell_time.py`](file:///c:/ДЗ/ASUS/src/analytics/dwell_time.py):
- Tracks how many cumulative seconds visitors spend inside each showcase polygon.
- Fires `dwell_detected` alert events when a visitor remains stationary beyond the zone's threshold ($1.5\text{s}$–$3.0\text{s}$).
- Computes **Attention Share (%)** across all active showcases to identify high-performing versus underperforming retail displays.

### 6. Queue Density & Cashier Alerting
- Continuously counts active visitors inside designated queue boundaries.
- Triggers instant `queue_overflow` telemetry events whenever occupancy exceeds maximum cashier capacity thresholds.

---

## ⚡ Hardware & NPU Acceleration (Hailo-10H)

Designed specifically for the **ASUS UGen300** accelerator card:

| Specification | Details |
|---|---|
| **NPU Processor** | Hailo-10H M.2 / PCIe AI Acceleration Module |
| **Peak Performance** | 40 TOPS (INT4) / 20 TOPS (INT8) |
| **On-Chip VRAM** | 8 GB LPDDR4X Dedicated Memory |
| **Thermal Envelope** | Fanless / Low-Power Edge Operation (~5W–15W) |
| **Model Format** | Compiled Hailo Executable Format (`.hef`) |
| **Runtime Library** | HailoRT Python SDK (`hailo_platform.pyhailort`) |

**Multi-Backend Fallback Hierarchy:**
1. **Hailo-10H NPU (`src/inference/hailo_infer.py`)** — Preferred for production deployment on ASUS hardware.
2. **ONNX Runtime (`src/inference/yolo_onnx.py`)** — Optimized CPU/GPU inference for standard x86/ARM devices.
3. **PyTorch Ultralytics (`YOLOv8n`)** — Plug-and-play development and evaluation baseline.

---

## 🚀 Quick Start Guide

### Prerequisites

- **OS:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or Debian 12.
- **Python:** 3.10, 3.11, or 3.12.
- **Camera / Video:** USB Webcam, RTSP IP Camera stream, or MP4 test video.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/FurionFarmila/SpaceMetrics-AI.git
cd SpaceMetrics-AI

# 2. Create and activate a Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Edge Pipeline

#### Option A: Run Full Web Dashboard & Live View (Recommended)
Launches the pipeline, web dashboard, live MJPEG streamer, and WebSocket server:
```bash
python main.py --serve --max-frames 0
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

#### Option B: Automated Demo Mode (Synthetic Video)
If no video file is provided, SpaceMetrics AI automatically synthesizes a simulated multi-agent retail video and runs the analytics:
```bash
python main.py --max-frames 250
```
Outputs generated in `output/`:
- `output/heatmap_total.png` (Overall floor presence)
- `output/heatmap_dwell.png` (Showcase attention hotspots)
- `output/heatmap_flow.png` (Transit pathways)
- `output/metrics_log.json` (Structured telemetry report)

#### Option C: Run on Custom Video or RTSP Stream
```bash
# Angled surveillance camera (foot anchor tracking):
python main.py --video "rtsp://admin:pass@192.168.1.50:554/stream1" --anchor bottom --serve

# Top-down ceiling camera (center anchor tracking):
python main.py --video path/to/topdown_video.mp4 --anchor center --serve
```

### Benchmark Datasets

Download long-duration realistic retail benchmark videos for testing:
```bash
python scripts/download_benchmarks.py
```
This downloads verified public domain CCTV / retail benchmark clips into `data/benchmarks/`.

---

## 🖥️ Web Dashboard & UI Guide

The web dashboard is structured into three intuitive tabs:

```
┌────────────────────────────────────────────────────────────────────────┐
│  📊 2D План зала и тепловые карты  │  🎥 Live View  │  ✏️ Разметка зон  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **📊 2D Floor Plan & Heatmaps (Tab 1):**
   - **Mode Selector:** Toggle between **Total Presence**, **Dwell (Attention)**, and **Transit Flow**.
   - **Colormaps:** Switch between **Turbo**, **Jet**, and **Inferno**.
   - **View Toggles:** Switch between **2D Floor Plan**, **Live Tracking Stream**, or **Split View** (side-by-side).
   - **Live Widgets:** Real-time KPIs (Active People, FPS, Total In/Out, Dwell Events), Showcase Attention Leaderboard, Tripwire Counts, and Live Tracks Table.

2. **🎥 Live View Stream (Tab 2):**
   - High-FPS annotated video feed directly from the sensor.
   - Live overlay toggle switches: Bounding Boxes, Trajectory Trails, Foot Anchors, Zones, and HUD.
   - Pause / Resume live feed button and single-frame snapshot capture.

3. **✏️ Visual Zone Markup (Tab 3):**
   - Click on the camera canvas to draw new polygons or lines:
     - **Витрина (Dwell Zone):** Click 3+ points around a product display or showcase.
     - **Касса (Queue Zone):** Click 3+ points around a checkout or queue area.
     - **Линия входа (Tripwire):** Click 2 points (Start and End) to establish a directional traffic barrier.
   - Click **"💾 Сохранить и применить в систему"** to save changes to [`configs/zones.json`](file:///c:/ДЗ/ASUS/configs/zones.json).

---

## 📡 REST API & WebSocket Documentation

| Method | Endpoint | Description | Response Type |
|---|---|---|---|
| `GET` | `/` | Web analytics dashboard, Live View, and visual markup editor | `text/html` |
| `GET` | `/api/metrics` | Latest anonymized edge telemetry (visitor count, FPS, tripwires, attention) | `application/json` |
| `GET` | `/api/heatmap?mode=total\|dwell\|flow` | 2D normalized density matrix ($60 \times 50$) of the floor grid | `application/json` |
| `GET` | `/api/heatmap/attention` | Showcase attention breakdown (seconds and share percentage) | `application/json` |
| `GET` | `/api/heatmap/image?mode=total\|dwell\|flow` | Dynamic PNG render of the floor plan with overlayed heatmap | `image/png` |
| `GET` | `/api/camera/stream` | Multi-part MJPEG video stream with real-time tracking overlays | `multipart/x-mixed-replace` |
| `GET` | `/api/camera/live_frame` | Single annotated JPEG frame snapshot | `image/jpeg` |
| `GET` | `/api/camera/frame` | Clean un-annotated camera JPEG frame for visual markup | `image/jpeg` |
| `GET` | `/api/camera/overlay_config` | Returns current visibility states for tracking overlay layers | `application/json` |
| `POST` | `/api/camera/overlay_config` | Dynamically updates layer toggles (boxes, trails, anchors, zones, HUD) | `application/json` |
| `GET` | `/api/zones/camera_projected` | Returns current zones projected onto camera pixel coordinates | `application/json` |
| `POST` | `/api/zones/save` | Receives new pixel markup, updates homography, and reloads engine | `application/json` |
| `GET` | `/api/config` | Retrieves sensor physical dimensions and zones configuration | `application/json` |
| `WS` | `/ws/metrics` | Real-time WebSocket telemetry push stream (~5 Hz) | `application/json` |

---

## 📁 Project Structure

```
SpaceMetrics-AI/
├── configs/
│   ├── config.yaml             # Main edge sensor parameters & thresholds
│   └── zones.json              # Homography calibration matrix & visual zones
├── data/
│   ├── benchmarks/             # Downloaded evaluation clips (.mp4)
│   ├── floor_plan.png          # 2D physical store layout background
│   └── sample_video.mp4        # Sample retail test video
├── output/
│   ├── .gitkeep                # Output directory placeholder
│   └── metrics_log.json        # Output telemetry log summary
├── scripts/
│   └── download_benchmarks.py  # Utility to fetch public retail benchmark datasets
├── src/
│   ├── analytics/
│   │   ├── counter.py          # Directional tripwire crossing & queue density
│   │   ├── dwell_time.py       # Dwell time tracking & attention share engine
│   │   └── heatmap.py          # Physical Gaussian splatting heatmap generator
│   ├── api/
│   │   └── app.py              # FastAPI server, WebSocket, and Web Dashboard UI
│   ├── capture/
│   │   └── video_stream.py     # Threaded multi-source video capture engine
│   ├── geometry/
│   │   └── homography.py       # Perspective matrix (Pixels ➔ Floor Meters)
│   ├── inference/
│   │   ├── base.py             # Inference interface & detection datatypes
│   │   ├── hailo_infer.py      # ASUS UGen300 / Hailo-10H NPU accelerator engine
│   │   └── yolo_onnx.py        # ONNX Runtime & PyTorch CPU/GPU fallback engine
│   ├── tracking/
│   │   └── tracker.py          # ByteTrack multi-object tracking integration
│   └── utils/
│       ├── demo_generator.py   # Synthetic retail multi-agent video generator
│       └── visualizer.py       # OpenCV visual overlay & Live View renderer
├── main.py                     # Primary pipeline orchestration CLI
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## ⚙️ Configuration Reference

### `configs/config.yaml`
```yaml
capture:
  source: "data/sample_video.mp4"   # RTSP URL, device index (0), or video path
  target_fps: 30
  width: 1280
  height: 720
  loop_video: true

geometry:
  floor_width_meters: 12.0          # Physical store width in meters
  floor_height_meters: 10.0         # Physical store depth in meters
  anchor_mode: "bottom"             # "bottom" (angled camera) or "center" (top-down)

analytics:
  heatmap:
    grid_width: 60                  # Floor grid resolution (X)
    grid_height: 50                 # Floor grid resolution (Y)
    footprint_radius_m: 0.45        # Gaussian footprint radius (meters)
    decay_half_life_sec: 0.0        # Heat cooling half-life (0 = accumulate)
  dwell_time:
    default_threshold_sec: 1.5      # Minimum time to register showcase interest
  queue:
    default_alert_threshold: 3      # Maximum people before queue alert
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Developed for the **ASUS UGen AI League — Battlefield Lightning (Lightweight Computing)**.
