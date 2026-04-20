<![CDATA[<div align="center">

<img src="https://img.shields.io/badge/TacticsAI-Real--time%20Football%20Analytics-1a1a2e?style=for-the-badge&logo=football&logoColor=white" alt="TacticsAI" />

# ⚽ TacticsAI — Real-time Football Analysis Platform

**AI-powered tactical analysis with live streaming, bird's-eye projection & formation detection**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF4B4B?style=flat-square)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📸 Preview

> *TacticsAI running on an FA Cup semi-final clip — Chelsea vs Manchester City*

The dashboard shows the annotated video feed (left), real-time bird's-eye pitch projection (top right), cumulative heatmap and tactical metrics (bottom right), all streaming live over WebSocket.

---

## 🧠 What is TacticsAI?

TacticsAI is a full-stack AI system that takes any football match video and, **in real time**, outputs:

- 🎯 **Player detection** using YOLOv8x (state-of-the-art object detection)
- 🔢 **Multi-player tracking** using ByteTrack (robust across occlusions)
- 🎽 **Automatic team classification** from jersey color (K-Means on HSV histograms)
- 🗺️ **Bird's-eye pitch projection** via perspective transform (camera → top-down 2D pitch)
- 🧭 **Formation detection** (4-4-2, 3-5-2, 1-2-1, etc.) with rolling-window smoothing
- 🔥 **Cumulative heatmaps** per team (where each team spends time on the pitch)
- 🕸️ **Pass network visualization** (who passes to whom, with frequency-weighted lines)
- 🏃 **Speed & distance tracking** (total metres run, max speed in km/h per player/team)
- 📐 **Voronoi space control** (pitch area "owned" by each team, coloured overlay)
- ⚡ **Pressing trigger detection** (when average team distance to ball drops below threshold)
- 👤 **Appearance-based Re-ID** (stable player IDs across tracker drops using HSV histograms)

All of this streams over **WebSocket** to a **Next.js 14 dashboard** — no page reloads, no polling.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Next.js 14 Frontend                  │
│              (localhost:3000)                        │
│                                                      │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ Annotated    │  │ Bird's-eye │  │  Heatmap /  │  │
│  │ Video Feed   │  │   Pitch    │  │  Metrics    │  │
│  │  (ref DOM)   │  │  (ref DOM) │  │ (React state│  │
│  └──────────────┘  └────────────┘  └─────────────┘  │
│              ↑ WebSocket (JSON + base64 JPEG)        │
└─────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend (localhost:8000)        │
│                                                      │
│  POST /api/upload  →  stores video, returns id       │
│  WS   /ws/{id}     →  streams analysis frames        │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │              ML Pipeline (src/)              │   │
│  │                                              │   │
│  │  VideoCapture → YOLOv8x → ByteTrack          │   │
│  │  → TeamClassifier → PerspectiveTransform     │   │
│  │  → FormationTracker → SpeedTracker           │   │
│  │  → PassNetwork → PressingDetector            │   │
│  │  → Voronoi → HeatmapEngine → AppearanceReID  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key design decisions:
- **Producer-consumer queue** (Python thread → asyncio coroutine) prevents the ML pipeline from blocking the WebSocket
- **Direct DOM updates via `useRef`** for image frames — bypasses React's re-render cycle entirely for 60fps-like visual updates
- **Two-pass video processing**: Pass 1 calibrates the team colour classifier on the first N frames; Pass 2 runs full analysis from the beginning
- **JPEG quality tuning**: Annotated frames at q=72, pitch view at q=80 — balanced between visual fidelity and WebSocket bandwidth
- **Heatmap throttling**: Sent every 5 frames instead of every frame to halve bandwidth without visible degradation

---

## 🚀 Getting Started

### Prerequisites

| Tool | Minimum Version |
|------|----------------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA GPU | Recommended (runs on CPU too, but ~5× slower) |
| pip | 23+ |

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/TacticsAI.git
cd TacticsAI
```

### 2. Python backend setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> On first run, YOLOv8x weights (~140 MB) will auto-download from Ultralytics.

### 3. Start the FastAPI backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend is now live at → `http://localhost:8000`  
API docs at → `http://localhost:8000/docs`

### 4. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Dashboard is now live at → `http://localhost:3000`

### 5. Upload & analyse

1. Open `http://localhost:3000`
2. Drag & drop (or click to upload) any football match video
3. Configure settings (frame skip, calibration frames, model weights)
4. Click **▶ Run Analysis**
5. Watch everything stream live

---

## 📁 Project Structure

```
TacticsAI/
│
├── api/
│   └── main.py                  # FastAPI app — upload, session, WebSocket
│
├── src/
│   ├── detection/
│   │   └── detector.py          # YOLOv8x wrapper (players + ball)
│   ├── tracking/
│   │   ├── tracker.py           # ByteTrack integration
│   │   └── reid.py              # Appearance-based Re-ID
│   ├── classification/
│   │   └── team_classifier.py   # K-Means jersey colour classifier
│   ├── pitch/
│   │   ├── calibration.py       # Auto-detect pitch corners (Hough + contours)
│   │   ├── transformer.py       # Perspective transform (camera → pitch coords)
│   │   └── renderer.py          # Top-down pitch canvas renderer
│   ├── tactics/
│   │   ├── formation.py         # K-Means formation detection + smoother
│   │   ├── heatmap.py           # Cumulative heatmap engine
│   │   ├── metrics.py           # Spatial metrics (width, depth, centroid)
│   │   ├── pass_network.py      # Pass count graph + renderer
│   │   ├── pressing.py          # Pressing trigger detector
│   │   ├── speed_tracker.py     # Per-player speed & distance accumulator
│   │   └── voronoi_viz.py       # Voronoi space control overlay
│   ├── annotation/
│   │   └── annotator.py         # Frame annotation (bounding boxes, trails)
│   └── pipeline.py              # Main generator: orchestrates all modules
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Main dashboard (all panels)
│   │   ├── layout.tsx           # Root layout
│   │   └── globals.css          # Dark theme design system
│   ├── hooks/
│   │   └── useAnalysis.ts       # WebSocket hook (ref-based image updates)
│   └── types/
│       └── analysis.ts          # TypeScript types
│
├── data/
│   └── tryolabs_demo.mp4        # Demo clip (see limitations below)
│
├── config.py                    # Global constants (canvas size, pitch dims)
├── requirements.txt
└── README.md
```

---

## ⚙️ Settings Explained

| Setting | Description | Recommended |
|---------|-------------|-------------|
| **Frame skip** | Process every N+1th frame. `2` = process every 3rd frame | `2` for GPU, `4` for CPU |
| **Max frames** | Cap total frames analysed. `0` = full video | `0` for full video |
| **Calibration frames** | Frames used to learn jersey colours before analysis begins | `60` |
| **Model weights** | `yolov8x` = most accurate, `yolov8m` = faster | `yolov8x` on GPU |
| **Device** | `CUDA` = GPU (fast), `CPU` = slow but always available | `CUDA` |

---

## 📊 Dashboard Panels

### Annotated Output
The original video frame with overlays:
- **Coloured bounding boxes** per team (detected by the K-Means jersey classifier)
- **Tracker IDs** for each player (ByteTrack stable across frames)
- **Ball detection** (yellow circle when visible)
- **Pass trajectory lines** (recent ball movement paths)
- **Ball Possession** overlay (time-weighted CHE vs MNC)

### Bird's-Eye View
Top-down 2D projection of the real pitch:
- **Coloured dots** per team (orange = Team A, green = Team B)
- **White outline ring** per dot for contrast on pitch background
- **Yellow ball dot** when ball is detectable
- **Voronoi overlay** — coloured zones showing which team "controls" each pitch area
- **Pass network lines** — thickness proportional to pass frequency between player pairs
- **Pressing border** — flashes when a team triggers pressing

### Cumulative Heatmap
Accumulates all frame data into a heat signature:
- Brighter = more time spent in that area
- Per-team colouring
- Updates every 5 frames

### Tactical Metrics
Per-team live stats:
- **Formation** (e.g. `4-2`, `3-3`, `1-2-1`)
- **Space control** (% of Voronoi area)
- **Width & Depth** (metres, spatial spread of team)
- **Distance run** (total metres since start)
- **Max speed** (km/h, team best)
- **Ball zone** (Defensive / Middle / Opposition Third)
- **Pressing events log** (timestamp, team, avg distance to ball)

---

## ⚠️ Demo Video Limitations

The included `data/tryolabs_demo.mp4` is a **short broadcast clip** (~58 seconds, 1467 frames).  
Some features appear limited because of this:

| Limitation | Cause | Full-match behaviour |
|---|---|---|
| **Formation often "Unknown"** for Team A | Camera shows only 2–3 players of Team A at once — not enough for K-Means clustering | Full match shows `4-4-2`, `4-3-3` etc. consistently |
| **Few dots on bird's-eye** in some frames | The camera zooms in tight → only 3–6 players in frame at any time | Wide-angle shots show all 10 outfield players |
| **Speed/distance numbers are low** | Only 58s of footage accumulates | Full 90-min match → typical 10–13 km/player |
| **Pass network is sparse** | Few pass events in a short clip | Dense graph after 20+ minutes |
| **Calibration uses same clip** | Perspective corners auto-detected from frame 0 | Same quality on any broadcast video |

> **For best results**: use a video with at least 5–10 minutes of footage, filmed from a fixed elevated broadcast camera. A tactical camera (side-on, full pitch visible) gives the best bird's-eye projection results.

---

## 🛠️ Technical Stack

### Backend
| Library | Purpose |
|---|---|
| `FastAPI` + `uvicorn` | REST API + WebSocket server |
| `ultralytics` (YOLOv8) | Player & ball detection |
| `supervision` | Detection post-processing, ByteTrack wrapper |
| `opencv-python` | Frame I/O, perspective transform, rendering |
| `scikit-learn` | K-Means (team classification, formation detection) |
| `numpy` | All numerical operations |
| `scipy` | Voronoi diagram computation |

### Frontend
| Library | Purpose |
|---|---|
| `Next.js 14` | React framework (App Router) |
| `TypeScript` | Type safety |
| `Vanilla CSS` | Custom dark design system |
| `useRef` DOM updates | Bypass React re-renders for video frames |

---

## 🗺️ Roadmap

- [x] YOLOv8 player & ball detection
- [x] ByteTrack multi-object tracking
- [x] K-Means jersey colour team classification
- [x] Perspective transform (camera → bird's-eye)
- [x] Formation detection with smoothing
- [x] Cumulative heatmaps
- [x] Pass network visualization
- [x] Speed & distance tracking
- [x] Voronoi space control
- [x] Pressing trigger detection
- [x] Appearance-based Re-ID
- [x] FastAPI WebSocket streaming backend
- [x] Next.js real-time dashboard
- [ ] Fine-tuned YOLOv8 on football-specific dataset
- [ ] Tactical camera auto-alignment (homography from pitch lines)
- [ ] Multi-match session history & comparison
- [ ] Export analysis to PDF / video clip
- [ ] Docker deployment

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ⚽ + 🐍 + ⚛️
</div>
]]>
