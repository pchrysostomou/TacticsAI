"""
Week 3 — Formation Detection

Determines team formation (4-4-2, 4-3-3, etc.) from the current
player positions projected onto the pitch canvas.

Method:
  1. Take all tracked players assigned to a team at this frame.
  2. Sort by pitch Y-coordinate (from own goal to opponent goal).
  3. K-Means cluster into N rows (defensors, midfielders, forwards).
  4. Count players in each cluster → produces formation string.
  5. Smooth over a rolling window to avoid per-frame jitter.

Week 4 extension: add pass-network graph from historical positions.
"""

from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

# ── Formation mapper ──────────────────────────────────────────────────────────
# Maps sorted layer counts (defenders → attackers) to formation string.
# Keys are tuples sorted high-Y to low-Y (i.e. playing vertical top→bottom).
_FORMATION_MAP: Dict[Tuple[int, ...], str] = {
    # 4-back systems
    (4, 4, 2): "4-4-2", (4, 3, 3): "4-3-3", (4, 2, 3, 1): "4-2-3-1",
    (4, 5, 1): "4-5-1", (4, 1, 4, 1): "4-1-4-1", (4, 4, 1, 1): "4-4-1-1",
    # 3-back systems
    (3, 5, 2): "3-5-2", (3, 4, 3): "3-4-3", (3, 3, 4): "3-3-4",
    # 5-back systems
    (5, 3, 2): "5-3-2", (5, 4, 1): "5-4-1",
    # Partial view (fewer players visible on camera)
    (4, 2): "4-2",   (3, 3): "3-3",   (2, 4): "2-4",
    (3, 2): "3-2",   (2, 3): "2-3",   (4, 1): "4-1",   (1, 4): "1-4",
    (2, 2): "2-2",   (3, 1): "3-1",   (1, 3): "1-3",   (2, 1): "2-1",
    (1, 2): "1-2",   (2, 2, 1): "2-2-1", (1, 2, 1): "1-2-1",
    (3, 2, 1): "3-2-1", (2, 3, 1): "2-3-1", (1, 3, 1): "1-3-1",
    (3, 1, 2): "3-1-2", (2, 1, 2): "2-1-2", (1, 1, 3): "1-1-3",
}
_UNKNOWN = "Unknown"

# Number of K-Means rows to attempt (try 4 → fall back to 3 → 2)
_ROW_ATTEMPTS = [4, 3, 2]


def _cluster_rows(
    y_coords: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Cluster Y-positions into n_rows horizontal lines."""
    km = KMeans(n_clusters=n_rows, n_init=5, max_iter=50, random_state=0)
    return km.fit_predict(y_coords.reshape(-1, 1))


def detect_formation(
    positions: List[Tuple[float, float]],
    pitch_height: int,
    team_plays_top_to_bottom: bool = True,
) -> str:
    """
    Return a formation string (e.g. '4-4-2') for one team's positions.
    Works with as few as 3 visible players (partial camera view).
    """
    # Need at least 3 players for any formation
    if len(positions) < 3:
        return _UNKNOWN

    pts = np.array(positions)
    y = pts[:, 1]

    # Exclude the goalkeeper (player closest to own goal line)
    if team_plays_top_to_bottom:
        gk_idx = int(np.argmin(y))
    else:
        gk_idx = int(np.argmax(y))
        y = pitch_height - y   # flip so smaller = own goal

    outfield_y = np.delete(y, gk_idx)

    if len(outfield_y) < 2:
        return _UNKNOWN

    # Try clustering into decreasing number of rows
    for n_rows in _ROW_ATTEMPTS:
        if len(outfield_y) < n_rows:
            continue
        labels = _cluster_rows(outfield_y, n_rows)

        # Get cluster centre Y for sorting rows (goal → attack)
        centres = {
            label: float(outfield_y[labels == label].mean())
            for label in range(n_rows)
        }
        # Sort rows by Y (small Y = defence side)
        sorted_rows = sorted(centres.keys(), key=lambda l: centres[l])
        counts = tuple(int((labels == row).sum()) for row in sorted_rows)

        formation = _FORMATION_MAP.get(counts)
        if formation:
            return formation

    return _UNKNOWN


# ── Smoothed formation tracker ────────────────────────────────────────────────

class FormationTracker:
    """
    Wraps detect_formation with a rolling-window vote smoothing so that the
    displayed formation doesn't flicker every frame.

    Usage::
        ft = FormationTracker(window=30)
        for frame in ...:
            smooth = ft.update(positions, pitch_height)
        ft.formation   # current stable formation
    """

    def __init__(self, n_teams: int = 2, window: int = 30) -> None:
        self.n_teams = n_teams
        self._windows: Dict[int, deque] = {
            t: deque(maxlen=window) for t in range(n_teams)
        }
        self._current: Dict[int, str] = {t: _UNKNOWN for t in range(n_teams)}

    def update(
        self,
        team_positions: Dict[int, List[Tuple[float, float]]],
        pitch_height: int,
    ) -> Dict[int, str]:
        """Update per-team formations and return the smoothed results."""
        for team_id, positions in team_positions.items():
            raw = detect_formation(positions, pitch_height)
            if raw != _UNKNOWN:
                self._windows[team_id].append(raw)
            if self._windows[team_id]:
                vote = Counter(self._windows[team_id]).most_common(1)[0][0]
                self._current[team_id] = vote
        return dict(self._current)

    @property
    def formations(self) -> Dict[int, str]:
        return dict(self._current)
