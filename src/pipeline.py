"""
Main video processing pipeline (Week 1 + Week 2).

Week 2 additions:
  - Two-pass design: calibration pass (jersey colour collection) then analysis.
  - K-Means team classification after calibration.
  - Perspective transform: camera → bird's-eye pitch coordinates.
  - Yields team_assignments, pitch_view, and calibration flag per frame.

Generator interface is fully backwards-compatible with Week 1.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    CALIBRATION_FRAMES,
    DEFAULT_MODEL,
    DEFAULT_SRC_POINTS,
    N_TEAMS,
    PITCH_CANVAS_H,
    PITCH_CANVAS_W,
)
from src.annotation.annotator import FrameAnnotator
from src.classification.team_classifier import TeamClassifier
from src.detection.detector import PlayerDetector
from src.pitch.renderer import PitchRenderer
from src.pitch.transformer import PerspectiveTransformer
from src.tactics.formation import FormationTracker
from src.tactics.heatmap import HeatmapEngine
from src.tactics.metrics import compute_metrics
from src.tactics.pass_network import PassNetwork
from src.tactics.pressing import PressingDetector
from src.tactics.speed_tracker import SpeedTracker
from src.tactics.voronoi_viz import render_voronoi
from src.tracking.reid import AppearanceReID
from src.tracking.tracker import PlayerTracker, TrackingResult


# ── Video metadata ────────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    info = {
        "fps":          cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_s"] = info["total_frames"] / info["fps"]
    cap.release()
    return info


# ── Core pipeline generator ───────────────────────────────────────────────────

def process_video(
    video_path: str,
    model_path: str = DEFAULT_MODEL,
    device: str = "cuda",
    skip_frames: int = 0,
    max_frames: Optional[int] = None,
    show_progress: bool = True,
    # Week 2 parameters
    src_points: Optional[List[List[float]]] = None,
    calibration_frames: int = CALIBRATION_FRAMES,
    n_teams: int = N_TEAMS,
    enable_team_classification: bool = True,
    enable_bird_eye: bool = True,
) -> Generator[dict, None, None]:
    """
    Week 1 + 2 + 3 + 4 pipeline generator.

    Performs two passes over the video when team classification is enabled:
      Pass 1 — calibration: collect jersey colour samples.
      Pass 2 — analysis:    detect → track → re-ID → classify → transform
                              → pass-net → pressing → speed → annotate.

    Args:
        video_path:               Input video file.
        model_path:               YOLOv8 weights (.pt).
        device:                   'cuda' or 'cpu'.
        skip_frames:              Process every (N+1)-th frame.
        max_frames:               Hard cap on processed frames.
        show_progress:            Print tqdm bar.
        src_points:               4 pitch corner points for perspective transform.
                                  None = use DEFAULT_SRC_POINTS.
        calibration_frames:       How many frames to collect jersey samples from.
        n_teams:                  Number of teams (K-Means clusters).
        enable_team_classification: Toggle team classification.
        enable_bird_eye:          Toggle bird's-eye view rendering.

    Yields:
        dict with keys:
            frame_idx         int
            timestamp         float
            annotated_frame   np.ndarray  (BGR)
            raw_frame         np.ndarray  (BGR)
            player_count      int
            ball_detected     bool
            tracking_result   TrackingResult
            team_assignments  Dict[int, int]   tracker_id → team_id
            team_counts       Dict[int, int]   team_id → player count
            team_colors_bgr   List[tuple]      BGR colour per team
            pitch_view        np.ndarray | None  Bird's-eye render
            calibrated        bool
            total_frames      int
            progress          float (0–1)
    """
    info = get_video_info(video_path)
    fps   = info["fps"]
    total = info["total_frames"]

    # ── Initialise components ─────────────────────────────────────────────────
    detector   = PlayerDetector(model_path=model_path, device=device)
    tracker    = PlayerTracker(fps=fps)
    annotator  = FrameAnnotator()
    classifier = TeamClassifier(n_teams=n_teams)
    renderer   = PitchRenderer(width=PITCH_CANVAS_W, height=PITCH_CANVAS_H)
    transformer = PerspectiveTransformer(
        src_points=src_points or DEFAULT_SRC_POINTS,
        dst_width=PITCH_CANVAS_W,
        dst_height=PITCH_CANVAS_H,
    )
    # Week 3
    formation_tracker = FormationTracker(n_teams=n_teams, window=30)
    heatmap_engine    = HeatmapEngine(pitch_w=PITCH_CANVAS_W, pitch_h=PITCH_CANVAS_H)
    # Week 4
    pass_network  = PassNetwork(n_teams=n_teams)
    speed_tracker = SpeedTracker(fps=fps, skip_frames=skip_frames)
    pressing_det  = PressingDetector(threshold_m=15.0, cooldown_s=3.0)
    reid_engine   = AppearanceReID()

    # ── Pass 1: Calibration ───────────────────────────────────────────────────
    if enable_team_classification and calibration_frames > 0:
        print(f"[Pipeline] Calibration: collecting jersey samples from "
              f"first {min(calibration_frames, total)} frames…")

        cap = cv2.VideoCapture(video_path)
        calib_count = 0

        pbar_calib = (
            tqdm(total=min(calibration_frames, total),
                 desc="Calibrating", unit="fr")
            if show_progress else None
        )

        while cap.isOpened() and calib_count < calibration_frames:
            ret, frame = cap.read()
            if not ret:
                break
            det = detector.detect(frame)
            for bbox in det.players.xyxy:
                classifier.collect(frame, bbox)
            calib_count += 1
            if pbar_calib:
                pbar_calib.update(1)

        if pbar_calib:
            pbar_calib.close()
        cap.release()

        fitted = classifier.fit()
        print(f"[Pipeline] Classifier {'fitted' if fitted else 'fit FAILED'} "
              f"on {classifier.n_samples} samples.")

    # ── Pass 2: Full analysis ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    tracker.reset()

    frame_idx = 0
    processed = 0
    pbar = (
        tqdm(total=total, desc="TacticsAI", unit="fr")
        if show_progress else None
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if pbar:
                pbar.update(1)

            if max_frames is not None and processed >= max_frames:
                break

            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            # ── Detection + Tracking ──────────────────────────────────────────
            det_result  = detector.detect(frame)
            track_result = tracker.update(det_result, frame_idx, fps)

            # ── W4: Re-identification ─────────────────────────────────────────────
            id_remap = reid_engine.update(frame, track_result.tracked_players)

            # ── Team Classification ─────────────────────────────────────────────
            team_assignments: Dict[int, int] = {}
            if enable_team_classification and classifier.is_fitted:
                tracked = track_result.tracked_players
                if tracked.tracker_id is not None:
                    for i, tid in enumerate(tracked.tracker_id):
                        team_id = classifier.classify(frame, tracked.xyxy[i], int(tid))
                        team_assignments[int(tid)] = team_id

            team_counts = classifier.team_counts(team_assignments)

            # ── Perspective Transform + Pitch View ──────────────────────────────
            pitch_view: Optional[np.ndarray] = None
            team_positions: Dict[int, List] = {t: [] for t in range(n_teams)}
            tracker_ids_by_team: Dict[int, List[int]] = {t: [] for t in range(n_teams)}
            team_by_id: Dict[int, int] = {}

            if enable_bird_eye and enable_team_classification and classifier.is_fitted:
                tracked = track_result.tracked_players
                if tracked.tracker_id is not None:
                    for i, tid in enumerate(tracked.tracker_id):
                        tid_int = int(tid)
                        canonical = reid_engine.canonical_id(tid_int)
                        feet_x, feet_y = transformer.get_feet(tracked.xyxy[i])
                        px, py = transformer.transform_point(feet_x, feet_y)
                        # Skip players whose transform puts them outside the pitch
                        # canvas (detected in stands, sidelines, etc.)
                        MARGIN = 80
                        if not (-MARGIN <= px <= PITCH_CANVAS_W + MARGIN and
                                -MARGIN <= py <= PITCH_CANVAS_H + MARGIN):
                            continue
                        t = team_assignments.get(tid_int, 0)
                        team_positions[t].append((float(px), float(py)))
                        tracker_ids_by_team[t].append(canonical)
                        team_by_id[canonical] = t

                ball_pitch: Optional[tuple] = None
                if det_result.ball is not None and len(det_result.ball) > 0:
                    bx1, by1, bx2, by2 = det_result.ball.xyxy[0]
                    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    bpx, bpy = transformer.transform_point(bcx, bcy)
                    ball_pitch = (bpx, bpy)

                # Speed tracking (W4)
                positions_by_id = {}
                for tid_int, (px, py) in zip(
                    [t for tl in tracker_ids_by_team.values() for t in tl],
                    [p for pl in team_positions.values() for p in pl],
                ):
                    positions_by_id[tid_int] = (px, py)
                speed_tracker.update(positions_by_id, frame_idx)

                # Pass network (W4)
                pass_network.update(ball_pitch, team_positions, tracker_ids_by_team)

                # Pressing detection (W4)
                pressing_state = pressing_det.update(
                    team_positions, ball_pitch,
                    timestamp=frame_idx / fps, frame_idx=frame_idx,
                )

                pitch_view = renderer.render(
                    team_positions=team_positions,
                    ball_position=ball_pitch,
                    team_colors_bgr=classifier.team_colors_bgr,
                    pass_network=pass_network,
                    tracker_ids_by_team=tracker_ids_by_team,
                    team_by_id=team_by_id,
                    show_voronoi=True,
                    speed_by_id=speed_tracker.all_data(),
                    pressing_state=pressing_state,
                )
            else:
                ball_pitch = None
                pressing_state: Dict[int, bool] = {}

            # ── Week 3: Formation + Heatmap + Metrics ─────────────────────────
            formations: Dict[int, str] = {}
            tactical_metrics: Dict = {}
            heatmap_view: Optional[np.ndarray] = None

            if enable_team_classification and classifier.is_fitted and team_positions:
                formations = formation_tracker.update(team_positions, PITCH_CANVAS_H)
                heatmap_engine.add_positions(team_positions)
                tactical_metrics = compute_metrics(
                    team_positions, ball_pitch,
                    pitch_w=PITCH_CANVAS_W, pitch_h=PITCH_CANVAS_H
                )
                if pitch_view is not None:
                    heatmap_view = heatmap_engine.render_both(
                        renderer._base,
                        team_colors_bgr=classifier.team_colors_bgr,
                    )

            # ── Annotation ────────────────────────────────────────────────────
            annotated = annotator.annotate(
                frame,
                track_result,
                team_assignments=team_assignments if enable_team_classification else None,
                team_colors_bgr=classifier.team_colors_bgr if classifier.is_fitted else None,
                pitch_view=pitch_view,
            )

            # ── Yield ─────────────────────────────────────────────────────────
            yield {
                "frame_idx":         frame_idx,
                "timestamp":         frame_idx / fps,
                "annotated_frame":   annotated,
                "raw_frame":         frame,
                "player_count":      len(track_result.tracked_players),
                "ball_detected":     det_result.ball is not None,
                "tracking_result":   track_result,
                "team_assignments":  team_assignments,
                "team_counts":       team_counts,
                "team_colors_bgr":   classifier.team_colors_bgr,
                "pitch_view":        pitch_view,
                "calibrated":        classifier.is_fitted,
                "total_frames":      total,
                "progress":          min(frame_idx / total, 1.0) if total > 0 else 0.0,
                # Week 3
                "formations":        formations,
                "heatmap_view":      heatmap_view,
                "tactical_metrics":  tactical_metrics,
                # Week 4
                "pressing_state":    pressing_state,
                "pressing_events":   pressing_det.recent_events(5),
                "speed_data":        speed_tracker.all_data(),
                "team_speed_summary": speed_tracker.team_summary(team_by_id),
                "pass_counts":       {t: pass_network.pass_counts(t) for t in range(n_teams)},
            }

            frame_idx += 1
            processed += 1

    finally:
        cap.release()
        if pbar:
            pbar.close()
