<div align="center">

# ⚽ TacticsAI

### Real-time AI Football Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-detection-FF4B4B?style=for-the-badge)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Upload a football match video → get live tactical analysis streamed to your browser in real time.**

</div>

---

## What is TacticsAI?

TacticsAI is a full-stack platform that runs a complete AI analysis pipeline on any football video and streams the results frame-by-frame to a custom Next.js dashboard — no page reloads, no polling.

The backend handles all the heavy lifting (detection, tracking, classification, tactical computation) over a FastAPI WebSocket. The frontend displays everything live using direct DOM updates via React refs, keeping the UI smooth even at high frame rates.

---

## Features

| Module | What it does |
|---|---|
| 🎯 **Player Detection** | YOLOv8x detects players, goalkeepers and the ball every frame |
| 🔢 **Multi-player Tracking** | ByteTrack assigns stable IDs across frames and handles occlusions |
| 🎽 **Team Classification** | K-Means on HSV jersey histograms automatically separates the two teams |
| 🗺️ **Bird's-eye View** | Perspective transform maps camera coordinates to a top-down 2D pitch |
| 🧭 **Formation Detection** | K-Means clusters player Y-positions into rows (4-4-2, 3-5-2, etc.) with rolling-window smoothing |
| 🔥 **Cumulative Heatmap** | Shows where each team spends time across the whole video |
| 🕸️ **Pass Network** | Frequency-weighted graph of which players exchange the ball |
| 🏃 **Speed & Distance** | Per-player: total metres run, max speed (km/h), avg speed |
| 📐 **Voronoi Space Control** | Coloured pitch zones showing which team "owns" each area |
| ⚡ **Pressing Detection** | Triggers when average team distance to ball drops below threshold |
| 👤 **Appearance Re-ID** | Cosine similarity on HSV histograms re-links player IDs after tracker drops |

---

## Architecture

```
┌──────────────────────────────────────────────┐
│             Next.js 14 — localhost:3000       │
│                                              │
│   Annotated Video │ Bird's-eye │ Heatmap     │
│   (ref DOM, no    │ (ref DOM)  │ / Metrics   │
│    React render)  │            │             │
│                                              │
│          ↑  WebSocket (JSON + JPEG base64)   │
└──────────────────────────────────────────────┘
                      │
┌──────────────────────────────────────────────┐
│            FastAPI — localhost:8000          │
│                                              │
│   POST /api/upload  →  store video           │
│   WS   /ws/{id}     →  stream frames         │
│                                              │
│   ┌──────────────────────────────────────┐   │
│   │           ML Pipeline (src/)         │   │
│   │                                      │   │
│   │  YOLOv8x → ByteTrack → K-Means      │   │
│   │  → PerspectiveTransform             │   │
│   │  → Formation / Heatmap / Speed       │   │
│   │  → PassNetwork / Pressing / Voronoi  │   │
│   │  → AppearanceReID                   │   │
│   └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

**Key decisions:**
- Producer-consumer queue keeps the ML thread and WebSocket async loop decoupled
- `useRef` direct image updates bypass React re-renders for smooth video
- Two-pass processing: calibration pass first, then full analysis from frame 0
- Heatmap sent every 5 frames (not every frame) to save bandwidth

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA GPU recommended (runs on CPU too, ~5× slower)

### 1 — Clone

```bash
git clone https://github.com/pchrysostomou/TacticsAI.git
cd TacticsAI
```

### 2 — Python backend

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> YOLOv8x weights (~140 MB) auto-download on first run.

### 3 — Start FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API → `http://localhost:8000`  
Swagger docs → `http://localhost:8000/docs`

### 4 — Start Next.js

```bash
cd frontend
npm install
npm run dev
```

Dashboard → `http://localhost:3000`

### 5 — Run analysis

1. Open `http://localhost:3000`
2. Upload any football match `.mp4`
3. Set frame skip, calibration frames, model (or leave defaults)
4. Click **▶ Run Analysis**
5. Watch everything stream live

---

## Project Structure

```
TacticsAI/
├── api/
│   └── main.py                   # FastAPI — upload endpoint + WebSocket stream
│
├── src/
│   ├── detection/detector.py     # YOLOv8x wrapper
│   ├── tracking/
│   │   ├── tracker.py            # ByteTrack
│   │   └── reid.py               # Appearance Re-ID
│   ├── classification/
│   │   └── team_classifier.py    # K-Means jersey colour
│   ├── pitch/
│   │   ├── calibration.py        # Auto-detect pitch corners
│   │   ├── transformer.py        # Perspective transform
│   │   └── renderer.py           # 2D pitch canvas renderer
│   ├── tactics/
│   │   ├── formation.py          # Formation detection + smoother
│   │   ├── heatmap.py            # Cumulative heatmap
│   │   ├── metrics.py            # Spatial metrics
│   │   ├── pass_network.py       # Pass graph
│   │   ├── pressing.py           # Pressing trigger
│   │   ├── speed_tracker.py      # Speed & distance
│   │   └── voronoi_viz.py        # Voronoi overlay
│   ├── annotation/annotator.py   # Bounding box + trail renderer
│   └── pipeline.py               # Main generator — orchestrates all modules
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx              # Dashboard (all panels)
│   │   ├── layout.tsx
│   │   └── globals.css           # Dark theme
│   ├── hooks/useAnalysis.ts      # WebSocket hook
│   └── types/analysis.ts
│
├── data/tryolabs_demo.mp4        # Demo clip (FA Cup semi-final)
├── config.py                     # Pitch dimensions, canvas size
└── requirements.txt
```

---

## Settings

| Setting | Description | Default |
|---|---|---|
| **Frame skip** | Process every N+1 frame (`2` = every 3rd) | `2` |
| **Max frames** | Stop after N frames (`0` = full video) | `0` |
| **Calibration frames** | Frames for jersey colour learning | `60` |
| **Model weights** | `yolov8x` (accurate) or `yolov8m` (fast) | `yolov8x` |
| **Device** | `CUDA` (GPU) or `CPU` | `CUDA` |

---

## Demo Video Notes

The included `data/tryolabs_demo.mp4` is a **~58 second FA Cup broadcast clip**.
This is intentionally small for quick testing. Because it is short and zoomed-in, some features look limited:

| What you see | Why | On a full match |
|---|---|---|
| Formation "Unknown" for one team | Camera shows only 2-3 players of that team at once | Stable 4-4-2 / 4-3-3 after ~2 minutes |
| Only 3-6 dots on bird's-eye | Tight camera angle, few players in frame | All 10 outfield players visible on tactical cam |
| Low km/speed totals | Only 58s of footage | ~10-13 km / player over 90 minutes |
| Sparse pass network | Few passes in a short clip | Dense graph after 20+ minutes |

> **Best results:** 5-10 min clip from a **fixed, elevated broadcast camera** or a dedicated tactical camera (side-on, full pitch visible).

---

## Tech Stack

**Backend:** Python · FastAPI · YOLOv8 · OpenCV · ByteTrack · scikit-learn · NumPy · SciPy

**Frontend:** Next.js 14 · TypeScript · Vanilla CSS · WebSocket

---

## Roadmap

- [x] YOLOv8 detection + ByteTrack tracking
- [x] Automatic team classification
- [x] Bird's-eye perspective transform
- [x] Formation detection
- [x] Cumulative heatmaps
- [x] Pass network
- [x] Speed & distance tracking
- [x] Voronoi space control
- [x] Pressing trigger detection
- [x] Appearance Re-ID
- [x] FastAPI WebSocket streaming
- [x] Next.js live dashboard
- [ ] Fine-tuned YOLOv8 (football-specific weights)
- [ ] Tactical camera homography from pitch lines
- [ ] Match history & comparison
- [ ] Export to PDF / annotated video
- [ ] Docker

---

<div align="center">
Built with ⚽ Python · FastAPI · YOLOv8 · Next.js
</div>
