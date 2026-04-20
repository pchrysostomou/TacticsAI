"""
Feature 1: Pass Network Visualization.

Detects passes by monitoring ball possession (ball proximity) and
recording possession transfers between tracked players.

Renders pass-network lines on the bird's-eye pitch canvas:
  - Line thickness  ∝  log(pass_count + 1)
  - Hub player (most passes) rendered with a larger dot
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ball must be within this many px of a player's feet to "own" possession
_POSSESSION_RADIUS_PX = 60

# Minimum passes before a connection is drawn (filter noise)
_MIN_DRAW_PASSES = 1


class PassNetwork:
    """
    Tracks ball possession and records passes between players.

    Usage::
        pn = PassNetwork(n_teams=2)

        # Each frame:
        pn.update(
            ball_pitch_pos,      # (px, py) on canvas, or None
            team_positions,      # {team_id: [(px,py),...]}
            tracker_ids_by_team, # {team_id: [tracker_id,...]}
        )

        # Render:
        img = pn.render(img, positions_by_id, team_by_id, team_colors_bgr)
    """

    def __init__(self, n_teams: int = 2) -> None:
        self.n_teams = n_teams
        # {team_id: {(id_a, id_b): count}}
        self._passes: Dict[int, Dict[Tuple[int, int], int]] = {
            t: defaultdict(int) for t in range(n_teams)
        }
        self._possessor: Optional[Tuple[int, int]] = None   # (team_id, tracker_id)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        ball_pitch: Optional[Tuple[float, float]],
        team_positions: Dict[int, List[Tuple[float, float]]],
        tracker_ids_by_team: Dict[int, List[int]],
    ) -> None:
        """Update possession state and record any pass that happened."""
        if ball_pitch is None:
            return

        bx, by = ball_pitch
        best_dist = float("inf")
        current_possessor: Optional[Tuple[int, int]] = None

        for team_id, positions in team_positions.items():
            tids = tracker_ids_by_team.get(team_id, [])
            for pos, tid in zip(positions, tids):
                px, py = pos
                dist = np.hypot(bx - px, by - py)
                if dist < best_dist:
                    best_dist = dist
                    current_possessor = (team_id, int(tid))

        if best_dist > _POSSESSION_RADIUS_PX:
            current_possessor = None

        # Detect pass: possession changed within the same team
        if (
            current_possessor is not None
            and self._possessor is not None
            and current_possessor != self._possessor
            and current_possessor[0] == self._possessor[0]   # same team
        ):
            team_id = current_possessor[0]
            from_id = self._possessor[1]
            to_id   = current_possessor[1]
            key = (min(from_id, to_id), max(from_id, to_id))   # undirected
            self._passes[team_id][key] += 1

        self._possessor = current_possessor

    # ── Query ─────────────────────────────────────────────────────────────────

    def pass_counts(self, team_id: int) -> Dict[Tuple[int, int], int]:
        return dict(self._passes[team_id])

    def hub_player(self, team_id: int) -> Optional[int]:
        """Return tracker_id with the highest total pass involvement."""
        counts = self._passes[team_id]
        if not counts:
            return None
        degree: Dict[int, int] = defaultdict(int)
        for (a, b), cnt in counts.items():
            degree[a] += cnt
            degree[b] += cnt
        return max(degree, key=degree.__getitem__)

    def reset(self) -> None:
        for t in range(self.n_teams):
            self._passes[t].clear()
        self._possessor = None

    # ── Render ────────────────────────────────────────────────────────────────

    def render(
        self,
        canvas: np.ndarray,
        pitch_positions_by_id: Dict[int, Tuple[float, float]],  # tracker_id → (px,py)
        team_by_id: Dict[int, int],                              # tracker_id → team_id
        team_colors_bgr: List[Tuple[int, int, int]],
        player_radius: int = 12,
    ) -> np.ndarray:
        """Overlay pass-network lines and hub-player markers."""
        out = canvas.copy()

        for team_id in range(self.n_teams):
            color  = team_colors_bgr[team_id % len(team_colors_bgr)]
            hub_id = self.hub_player(team_id)
            counts = self._passes[team_id]

            if not counts:
                continue

            max_cnt = max(counts.values(), default=1)

            for (id_a, id_b), cnt in counts.items():
                if cnt < _MIN_DRAW_PASSES:
                    continue
                if id_a not in pitch_positions_by_id or id_b not in pitch_positions_by_id:
                    continue
                pa = tuple(int(v) for v in pitch_positions_by_id[id_a])
                pb = tuple(int(v) for v in pitch_positions_by_id[id_b])

                thickness = max(1, int(np.log1p(cnt) / np.log1p(max_cnt) * 6))
                alpha_line = 0.65

                # Draw semi-transparent line
                overlay = out.copy()
                cv2.line(overlay, pa, pb, color, thickness, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha_line, out, 1 - alpha_line, 0, out)

                # Pass count label at midpoint
                mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
                cv2.putText(out, str(cnt), mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # Hub player — larger dot with glow
            if hub_id is not None and hub_id in pitch_positions_by_id:
                hx, hy = (int(v) for v in pitch_positions_by_id[hub_id])
                cv2.circle(out, (hx, hy), player_radius + 8, color, 3)
                cv2.circle(out, (hx, hy), player_radius + 8, (255, 255, 255), 1)

        return out
