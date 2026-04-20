"""
Feature 3: Per-player speed & distance tracking.

Accumulates total distance run (metres) for each tracked player and
derives instantaneous / average / max speed.

Real-world scaling uses the pitch canvas pixel-to-metre ratio derived
from FIFA standard dimensions (105m × 68m) mapped to the canvas.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from config import PITCH_CANVAS_H, PITCH_CANVAS_W, PITCH_REAL_H_M, PITCH_REAL_W_M

# Average pixel-to-metre factor (mean of both axes)
_PX_TO_M = (PITCH_REAL_W_M / PITCH_CANVAS_W + PITCH_REAL_H_M / PITCH_CANVAS_H) / 2


class PlayerSpeedData:
    """Accumulated metrics for a single player."""

    __slots__ = (
        "total_dist_m", "max_speed_ms", "_speed_sum", "_speed_n",
        "_last_pos", "_last_frame",
    )

    def __init__(self) -> None:
        self.total_dist_m: float = 0.0
        self.max_speed_ms: float = 0.0
        self._speed_sum:   float = 0.0
        self._speed_n:     int   = 0
        self._last_pos:    Optional[Tuple[float, float]] = None
        self._last_frame:  Optional[int] = None

    @property
    def avg_speed_ms(self) -> float:
        return self._speed_sum / self._speed_n if self._speed_n else 0.0

    def to_dict(self) -> dict:
        return {
            "total_dist_m":  round(self.total_dist_m, 1),
            "max_speed_ms":  round(self.max_speed_ms, 1),
            "avg_speed_ms":  round(self.avg_speed_ms, 1),
            "max_speed_kmh": round(self.max_speed_ms * 3.6, 1),
            "avg_speed_kmh": round(self.avg_speed_ms * 3.6, 1),
        }


class SpeedTracker:
    """
    Tracks speed and distance for every active player.

    Usage::
        st = SpeedTracker(fps=25)
        # each frame:
        st.update(pitch_positions_by_id, frame_idx)
        data = st.all_data()     # {tracker_id: {...}}
        team_summary = st.team_summary(team_by_id)
    """

    def __init__(self, fps: float = 25.0, skip_frames: int = 0) -> None:
        self.fps          = fps
        self.frame_step   = skip_frames + 1          # effective time per frame
        self._data: Dict[int, PlayerSpeedData] = {}

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        pitch_positions_by_id: Dict[int, Tuple[float, float]],
        frame_idx: int,
    ) -> None:
        """Update metrics for all players visible in this frame."""
        dt = self.frame_step / self.fps   # seconds per processed frame

        for tracker_id, pos in pitch_positions_by_id.items():
            if tracker_id not in self._data:
                self._data[tracker_id] = PlayerSpeedData()

            pd = self._data[tracker_id]

            if pd._last_pos is not None:
                dx = pos[0] - pd._last_pos[0]
                dy = pos[1] - pd._last_pos[1]
                dist_px = np.hypot(dx, dy)
                dist_m  = dist_px * _PX_TO_M

                # Clamp implausible jumps (tracker glitch: >12 m/s = ~43 km/h)
                speed_ms = dist_m / max(dt, 1e-6)
                if speed_ms < 12.0:
                    pd.total_dist_m += dist_m
                    pd._speed_sum   += speed_ms
                    pd._speed_n     += 1
                    if speed_ms > pd.max_speed_ms:
                        pd.max_speed_ms = speed_ms

            pd._last_pos   = pos
            pd._last_frame = frame_idx

        # Evict players not seen for >300 frames (they've left the pitch)
        stale = [
            tid for tid, pd in self._data.items()
            if pd._last_frame is not None and (frame_idx - pd._last_frame) > 300
        ]
        for tid in stale:
            del self._data[tid]

    # ── Query ─────────────────────────────────────────────────────────────────

    def get(self, tracker_id: int) -> Optional[PlayerSpeedData]:
        return self._data.get(tracker_id)

    def all_data(self) -> Dict[int, dict]:
        return {tid: pd.to_dict() for tid, pd in self._data.items()}

    def team_summary(self, team_by_id: Dict[int, int]) -> Dict[int, dict]:
        """Aggregate per-team: total distance, top sprinter, avg speed."""
        from collections import defaultdict
        team_dists: Dict[int, list] = defaultdict(list)
        team_speeds: Dict[int, list] = defaultdict(list)
        team_max: Dict[int, float] = defaultdict(float)
        team_top: Dict[int, Optional[int]] = {}

        for tid, pd in self._data.items():
            t = team_by_id.get(tid, -1)
            if t < 0:
                continue
            team_dists[t].append(pd.total_dist_m)
            team_speeds[t].append(pd.avg_speed_ms)
            if pd.max_speed_ms > team_max[t]:
                team_max[t] = pd.max_speed_ms
                team_top[t] = tid

        result = {}
        for team_id in set(team_by_id.values()):
            dists  = team_dists.get(team_id, [0])
            speeds = team_speeds.get(team_id, [0])
            result[team_id] = {
                "total_dist_m":    round(sum(dists), 0),
                "avg_dist_m":      round(np.mean(dists), 0) if dists else 0,
                "team_max_spd_ms": round(team_max.get(team_id, 0), 1),
                "team_max_spd_kmh": round(team_max.get(team_id, 0) * 3.6, 1),
                "top_sprinter_id": team_top.get(team_id),
            }
        return result

    def reset(self) -> None:
        self._data.clear()
