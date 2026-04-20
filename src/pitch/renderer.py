"""
Updated PitchRenderer — supports all Week 4 overlays:
  - Pass network (lines + hub dot)
  - Voronoi control zones
  - Player dots with speed badge
  - Pressing highlight border
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.tactics.pass_network import PassNetwork
from src.tactics.voronoi_viz import render_voronoi

_GREEN_LIGHT = (55, 130, 55)
_GREEN_DARK  = (42, 108, 42)
_WHITE       = (255, 255, 255)
_BALL_COLOR  = (0, 230, 255)
_SHADOW      = (0, 0, 0)
_PRESSING_BORDER = (0, 60, 255)    # red border when pressing active


class PitchRenderer:
    """Renders a top-down football pitch with all analytical overlays."""

    def __init__(self, width: int = 1040, height: int = 680) -> None:
        self.w = width
        self.h = height
        self._base = self._build_pitch()

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        team_positions: Dict[int, List[Tuple[float, float]]],
        ball_position: Optional[Tuple[float, float]] = None,
        team_colors_bgr: Optional[List[Tuple[int, int, int]]] = None,
        # Week 4 optional overlays
        pass_network: Optional[PassNetwork] = None,
        tracker_ids_by_team: Optional[Dict[int, List[int]]] = None,
        team_by_id: Optional[Dict[int, int]] = None,
        show_voronoi: bool = False,
        speed_by_id: Optional[Dict[int, dict]] = None,
        pressing_state: Optional[Dict[int, bool]] = None,
    ) -> np.ndarray:
        """
        Render the pitch canvas with all requested overlays.

        Args:
            team_positions:       {team_id: [(px,py),...]}
            ball_position:        (px,py) or None.
            team_colors_bgr:      BGR per team.
            pass_network:         PassNetwork instance (optional).
            tracker_ids_by_team:  {team_id: [tracker_id,...]} for pass network.
            team_by_id:           {tracker_id: team_id} for Voronoi / pass net.
            show_voronoi:         Draw Voronoi zones.
            speed_by_id:          {tracker_id: {"max_speed_kmh": float}} for badge.
            pressing_state:       {team_id: bool} — highlight border if pressing.
        """
        default_colors: List[Tuple[int, int, int]] = [(255, 60, 30), (30, 200, 50)]
        colors = team_colors_bgr or default_colors
        # Use vivid accent colours for player dots (jersey-detected colors can be
        # too pale/washed out; these are always clearly visible on green pitch)
        DOT_COLORS: List[Tuple[int, int, int]] = [(245, 92, 30), (30, 210, 60)]
        canvas = self._base.copy()

        # Build pitch_positions_by_id for overlays
        positions_by_id: Dict[int, Tuple[float, float]] = {}
        if tracker_ids_by_team and team_positions:
            for team_id, positions in team_positions.items():
                tids = tracker_ids_by_team.get(team_id, [])
                for pos, tid in zip(positions, tids):
                    positions_by_id[int(tid)] = pos

        # ── Voronoi ───────────────────────────────────────────────────────────
        if show_voronoi and positions_by_id and team_by_id:
            canvas = render_voronoi(
                canvas, positions_by_id, team_by_id, colors
            )

        # ── Pass network ──────────────────────────────────────────────────────
        if pass_network is not None and positions_by_id and team_by_id:
            canvas = pass_network.render(canvas, positions_by_id, team_by_id, colors)

        # ── Players ───────────────────────────────────────────────────────────
        for team_id, positions in team_positions.items():
            color    = colors[team_id % len(colors)]      # jersey color (for Voronoi)
            dot_col  = DOT_COLORS[team_id % len(DOT_COLORS)]  # vivid dot color
            tids  = (tracker_ids_by_team or {}).get(team_id, [None] * len(positions))
            for pos, tid in zip(positions, tids):
                cx = int(np.clip(pos[0], 4, self.w - 4))
                cy = int(np.clip(pos[1], 4, self.h - 4))
                # Shadow
                cv2.circle(canvas, (cx + 2, cy + 2), 14, _SHADOW, -1)
                # Filled vivid-colour dot (always visible)
                cv2.circle(canvas, (cx,     cy),     14, dot_col, -1)
                # White outer ring (contrast)
                cv2.circle(canvas, (cx,     cy),     14, _WHITE,   2)
                # Small white centre dot for precision
                cv2.circle(canvas, (cx,     cy),      3, _WHITE,  -1)

                # Speed badge — km/h inside dot
                if tid is not None and speed_by_id and int(tid) in speed_by_id:
                    spd = speed_by_id[int(tid)].get("max_speed_kmh", 0)
                    if spd > 0:
                        cv2.putText(canvas, f"{spd:.0f}", (cx - 9, cy + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                    _WHITE, 1, cv2.LINE_AA)

        # ── Ball ──────────────────────────────────────────────────────────────
        if ball_position is not None:
            bx = int(np.clip(ball_position[0], 2, self.w - 2))
            by = int(np.clip(ball_position[1], 2, self.h - 2))
            cv2.circle(canvas, (bx + 1, by + 1), 7, _SHADOW,     -1)
            cv2.circle(canvas, (bx,     by),      7, _BALL_COLOR, -1)
            cv2.circle(canvas, (bx,     by),      7, _SHADOW,      2)

        # ── Pressing border ───────────────────────────────────────────────────
        if pressing_state:
            pressing_teams = [t for t, v in pressing_state.items() if v]
            if pressing_teams:
                border_color = colors[pressing_teams[0] % len(colors)]
                # Pulsing red border
                cv2.rectangle(canvas, (3, 3), (self.w - 3, self.h - 3),
                               border_color, 4)
                label = f"PRESS! Team {'AB'[pressing_teams[0]]}"
                cv2.putText(canvas, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, border_color, 2, cv2.LINE_AA)

        return canvas

    # ── Pitch drawing ─────────────────────────────────────────────────────────

    def _build_pitch(self) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = _GREEN_LIGHT

        sw = self.w // 10
        for i in range(0, 10, 2):
            img[:, i * sw:(i + 1) * sw] = _GREEN_DARK

        p = 40
        W, H = self.w, self.h
        mx, my = W // 2, H // 2

        def rect(x0, y0, x1, y1, t=2):
            cv2.rectangle(img, (x0, y0), (x1, y1), _WHITE, t)

        def line(x0, y0, x1, y1, t=2):
            cv2.line(img, (x0, y0), (x1, y1), _WHITE, t)

        def circle(cx, cy, r, t=2):
            cv2.circle(img, (cx, cy), r, _WHITE, t)

        def dot(cx, cy, r=4):
            cv2.circle(img, (cx, cy), r, _WHITE, -1)

        rect(p, p, W - p, H - p)
        line(mx, p, mx, H - p)
        circle(mx, my, int(H * 0.136))
        dot(mx, my)

        pen_x = int(W * 0.157)
        pen_h = int(H * 0.593)
        pt    = (H - pen_h) // 2
        rect(p,             pt, p + pen_x,         pt + pen_h)
        rect(W - p - pen_x, pt, W - p,             pt + pen_h)

        six_x = int(W * 0.057)
        six_h = int(H * 0.265)
        st    = (H - six_h) // 2
        rect(p,             st, p + six_x,         st + six_h)
        rect(W - p - six_x, st, W - p,             st + six_h)

        sp = int(W * 0.105)
        dot(p + sp,     my)
        dot(W - p - sp, my)

        ar = int(H * 0.136)
        cv2.ellipse(img, (p + sp,     my), (ar, ar), 0, -54,  54,  _WHITE, 2)
        cv2.ellipse(img, (W - p - sp, my), (ar, ar), 0,  126, 234, _WHITE, 2)

        gh = int(H * 0.107)
        gd = int(W * 0.019)
        gt = (H - gh) // 2
        rect(p - gd,    gt, p,          gt + gh)
        rect(W - p,     gt, W - p + gd, gt + gh)

        return img
