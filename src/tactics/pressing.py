"""
Feature 4: Pressing Trigger Detection.

Detects the moment a team initiates a high-press: their players'
average distance to the ball drops below a configurable threshold
(default 15 m in real pitch space).

Records events with timestamp, team, and average distance.
The pressing state (active / not) is also exposed per team so the
dashboard can highlight the current frame.

Usage::
    pd = PressingDetector(threshold_m=15.0)

    # Each frame:
    state = pd.update(
        team_positions,     # {team_id: [(px,py),...]}
        ball_pitch_pos,     # (px, py) or None
        timestamp,          # frame time in seconds
    )
    # state = {team_id: bool}  True = currently pressing
    # pd.events            # list of recorded pressing triggers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import PITCH_CANVAS_W, PITCH_REAL_W_M

_PX_TO_M = PITCH_REAL_W_M / PITCH_CANVAS_W

# Hysteresis: must stay below threshold for N consecutive frames to trigger
_HYSTERESIS_FRAMES = 5


@dataclass
class PressingEvent:
    timestamp:   float
    team_id:     int
    avg_dist_m:  float
    frame_idx:   int


class PressingDetector:
    """
    Detects pressing triggers based on average team-to-ball distance.

    Args:
        threshold_m:  Distance (metres) below which pressing is active.
        cooldown_s:   Minimum seconds between logged events (de-bounce).
    """

    def __init__(self, threshold_m: float = 15.0, cooldown_s: float = 3.0) -> None:
        self.threshold_m  = threshold_m
        self.cooldown_s   = cooldown_s
        self.events: List[PressingEvent] = []
        self._pressing: Dict[int, bool]         = {}
        self._below_count: Dict[int, int]       = {}   # frames consecutively below threshold
        self._last_event_ts: Dict[int, float]   = {}

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        team_positions: Dict[int, List[Tuple[float, float]]],
        ball_pitch: Optional[Tuple[float, float]],
        timestamp: float,
        frame_idx: int = 0,
    ) -> Dict[int, bool]:
        """
        Returns {team_id: is_pressing (bool)}.
        Also appends to self.events when a new pressing trigger fires.
        """
        if ball_pitch is None:
            return dict(self._pressing)

        bx, by = ball_pitch

        for team_id, positions in team_positions.items():
            if not positions:
                self._pressing[team_id] = False
                continue

            pts = np.array(positions, dtype=np.float32)
            dists_px = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
            avg_dist_m = float(dists_px.mean() * _PX_TO_M)

            is_below = avg_dist_m < self.threshold_m
            self._below_count[team_id] = self._below_count.get(team_id, 0)

            if is_below:
                self._below_count[team_id] += 1
            else:
                self._below_count[team_id] = 0

            was_pressing = self._pressing.get(team_id, False)
            now_pressing = self._below_count[team_id] >= _HYSTERESIS_FRAMES
            self._pressing[team_id] = now_pressing

            # Log a new event at rising edge, respecting cooldown
            if now_pressing and not was_pressing:
                last_ts = self._last_event_ts.get(team_id, -999)
                if timestamp - last_ts >= self.cooldown_s:
                    self.events.append(PressingEvent(
                        timestamp  = round(timestamp, 2),
                        team_id    = team_id,
                        avg_dist_m = round(avg_dist_m, 1),
                        frame_idx  = frame_idx,
                    ))
                    self._last_event_ts[team_id] = timestamp

        return dict(self._pressing)

    # ── Query ─────────────────────────────────────────────────────────────────

    def is_pressing(self, team_id: int) -> bool:
        return self._pressing.get(team_id, False)

    def recent_events(self, n: int = 10) -> List[PressingEvent]:
        return self.events[-n:]

    def reset(self) -> None:
        self.events.clear()
        self._pressing.clear()
        self._below_count.clear()
        self._last_event_ts.clear()
