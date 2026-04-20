"""
ByteTrack player tracker (via supervision).

Assigns a persistent integer ID to each detected player across frames,
even through brief occlusions (players overlapping or leaving frame).

State (TrackedPlayer) accumulates centroid positions across the video,
which will feed into heatmap generation in Week 3.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv

from config import (
    DEFAULT_FPS,
    LOST_TRACK_BUFFER,
    MINIMUM_MATCHING_THRESHOLD,
    TRACK_ACTIVATION_THRESHOLD,
)
from src.detection.detector import DetectionResult


# ── Per-player state ──────────────────────────────────────────────────────────

@dataclass
class TrackedPlayer:
    """
    Persistent state for one tracked player across all frames.

    positions: list of (cx, cy) pixel centroids — used for heatmaps (Week 3).
    team_id:   filled in Week 2 after K-Means jersey classification.
    """

    tracker_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    last_bbox: Optional[np.ndarray] = None
    team_id: Optional[int] = None           # 0 = Team A, 1 = Team B (Week 2)

    def record_position(self, bbox: np.ndarray) -> None:
        """Append centroid from xyxy bounding box."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.positions.append((float(cx), float(cy)))
        self.last_bbox = bbox.copy()

    @property
    def current_position(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    @property
    def position_history(self) -> np.ndarray:
        """Return positions as (N, 2) numpy array."""
        return np.array(self.positions, dtype=np.float32)


# ── Frame-level tracking result ───────────────────────────────────────────────

@dataclass
class TrackingResult:
    """All tracking data for a single processed frame."""

    tracked_players: sv.Detections              # Players with .tracker_id set
    ball: Optional[sv.Detections]               # Ball detection (or None)
    player_states: Dict[int, TrackedPlayer]     # Full history keyed by track ID
    frame_idx: int
    timestamp: float                            # Seconds from video start


# ── Tracker ───────────────────────────────────────────────────────────────────

class PlayerTracker:
    """
    Wraps supervision.ByteTracker for multi-object player tracking.

    Lifecycle::
        tracker = PlayerTracker(fps=25.0)
        for frame_detections in detections_stream:
            result = tracker.update(frame_detections, frame_idx, fps)
        tracker.reset()   # between different videos
    """

    def __init__(self, fps: float = DEFAULT_FPS) -> None:
        self._fps = fps
        self._byte_tracker = sv.ByteTrack(
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=LOST_TRACK_BUFFER,
            minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
            frame_rate=int(fps),
        )
        self.player_states: Dict[int, TrackedPlayer] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        detection_result: DetectionResult,
        frame_idx: int,
        fps: float,
    ) -> TrackingResult:
        """
        Update tracker with detections from a new frame.

        Args:
            detection_result: Output of PlayerDetector.detect().
            frame_idx:        Zero-based frame index.
            fps:              Video FPS (for timestamp calculation).

        Returns:
            TrackingResult containing tracked detections + full player state.
        """
        # ByteTrack assigns tracker_id to each detection
        tracked = self._byte_tracker.update_with_detections(
            detection_result.players
        )

        # Accumulate position history for each active track
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                if tid not in self.player_states:
                    self.player_states[tid] = TrackedPlayer(tracker_id=tid)
                self.player_states[tid].record_position(tracked.xyxy[i])

        return TrackingResult(
            tracked_players=tracked,
            ball=detection_result.ball,
            player_states=self.player_states,
            frame_idx=frame_idx,
            timestamp=frame_idx / fps if fps > 0 else 0.0,
        )

    def reset(self) -> None:
        """Clear all state — call between different video files."""
        self._byte_tracker.reset()
        self.player_states.clear()

    @property
    def active_track_count(self) -> int:
        return len(self.player_states)
