"""TacticsAI FastAPI Backend — optimised for streaming."""

import asyncio
import base64
import threading
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _to_py(v):
    """Convert numpy scalar to Python native type (JSON-safe)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

app = FastAPI(title="TacticsAI API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = Path("tmp_uploads")
TMP_DIR.mkdir(exist_ok=True)
_sessions: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode(frame: Optional[np.ndarray], quality: int = 70) -> Optional[str]:
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8") if ok else None


def _safe(d: dict) -> dict:
    return {str(k): v for k, v in d.items()}


def _make_payload(result: dict, send_heatmap: bool = True) -> dict:
    tm       = result.get("tactical_metrics") or {}
    sc       = tm.get("space_control", {})
    pi       = tm.get("pressing_intensity", {})
    teams_tm = tm.get("teams", {})
    ball_zone = tm.get("ball_zone", "—")

    teams_out: dict = {}
    for tid, t in teams_tm.items():
        teams_out[str(tid)] = {
            "width_m":  t.get("width_m", 0),
            "depth_m":  t.get("depth_m", 0),
            "cohesion": t.get("cohesion_m", 0),
        }

    spd_out: dict = {}
    for tid, s in (result.get("team_speed_summary") or {}).items():
        spd_out[str(tid)] = {
            "total_dist_m":     _to_py(s.get("total_dist_m", 0)),
            "team_max_spd_kmh": _to_py(s.get("team_max_spd_kmh", 0)),
            "top_sprinter_id":  str(s.get("top_sprinter_id") or ""),
        }

    evts = [
        {"timestamp": e.timestamp, "team_id": e.team_id, "avg_dist_m": e.avg_dist_m}
        for e in (result.get("pressing_events") or [])
    ]

    return {
        "type":            "frame",
        "frame_idx":       _to_py(result["frame_idx"]),
        "timestamp":       round(_to_py(result["timestamp"]), 2),
        "progress":        round(_to_py(result["progress"]), 4),
        "total_frames":    _to_py(result["total_frames"]),
        "player_count":    _to_py(result["player_count"]),
        "ball_detected":   bool(result["ball_detected"]),
        # Images — heatmap sent only every N frames to save bandwidth
        "annotated":       _encode(result.get("annotated_frame"), 72),
        "pitch_view":      _encode(result.get("pitch_view"),      80),
        "heatmap_view":    _encode(result.get("heatmap_view"),    65) if send_heatmap else None,
        # Lightweight metadata
        "team_counts":     _safe(result.get("team_counts") or {}),
        "formations":      _safe(result.get("formations")  or {}),
        "pressing_state":  _safe(result.get("pressing_state") or {}),
        "pressing_events": evts,
        "space_control":   _safe(sc),
        "press_intensity": _safe(pi),
        "teams_spatial":   teams_out,
        "ball_zone":       ball_zone,
        "team_speed":      spd_out,
    }


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "TacticsAI API"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    session_id = uuid.uuid4().hex[:12]
    dest = TMP_DIR / f"{session_id}_{file.filename}"
    dest.write_bytes(await file.read())

    cap = cv2.VideoCapture(str(dest))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    info = {"fps": fps, "total_frames": total, "width": w, "height": h,
            "duration_s": round(total / fps, 2), "filename": file.filename}
    _sessions[session_id] = {"path": str(dest), "info": info}
    return {"session_id": session_id, "info": info}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    sess = _sessions.get(session_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return sess["info"]


# ── WebSocket streaming endpoint ──────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def ws_analyze(
    websocket:          WebSocket,
    session_id:         str,
    skip_frames:        int = Query(2),
    max_frames:         int = Query(300),
    calibration_frames: int = Query(60),
    model_path:         str = Query("yolov8x.pt"),
    device:             str = Query("cuda"),
):
    await websocket.accept()

    sess = _sessions.get(session_id)
    if not sess:
        await websocket.send_json({"type": "error", "message": "session not found"})
        await websocket.close()
        return

    video_path = sess["path"]
    loop       = asyncio.get_event_loop()
    # Large queue — producer never blocks on slow consumer
    q: asyncio.Queue = asyncio.Queue(maxsize=24)

    # ── Auto-calibrate from first frame ──────────────────────────────
    src_points = None
    try:
        from src.pitch.calibration import auto_detect_pitch_corners
        cap0 = cv2.VideoCapture(video_path)
        ret0, frm0 = cap0.read()
        cap0.release()
        if ret0:
            src_points = auto_detect_pitch_corners(frm0)
            await websocket.send_json({"type": "calibration", "src_points": src_points})
    except Exception:
        pass

    # ── Producer thread ───────────────────────────────────────────────
    frame_n = 0

    def _producer():
        nonlocal frame_n
        from src.pipeline import process_video
        try:
            for result in process_video(
                video_path,
                model_path=model_path,
                device=device,
                skip_frames=skip_frames,
                max_frames=max_frames if max_frames > 0 else None,
                show_progress=False,
                src_points=src_points,
                calibration_frames=calibration_frames,
                enable_team_classification=True,
                enable_bird_eye=True,
            ):
                frame_n += 1
                # Heatmap every 5 frames — it's slow-changing and heavy
                payload = _make_payload(result, send_heatmap=(frame_n % 5 == 0))
                asyncio.run_coroutine_threadsafe(q.put(payload), loop).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "error", "message": str(exc)}), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

    threading.Thread(target=_producer, daemon=True).start()

    # ── Consumer (async) ──────────────────────────────────────────────
    try:
        while True:
            payload = await asyncio.wait_for(q.get(), timeout=120.0)
            if payload is None:
                await websocket.send_json({"type": "done"})
                break
            await websocket.send_json(payload)
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "pipeline timeout"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        # ── Cleanup: delete uploaded video + remove from session store ──
        try:
            await websocket.close()
        except Exception:
            pass
        try:
            Path(video_path).unlink(missing_ok=True)
            _sessions.pop(session_id, None)
        except Exception:
            pass
