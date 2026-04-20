"""
Frame annotator — Week 1 + Week 2.

Week 1:  bounding boxes, player IDs, ball, HUD overlay.
Week 2:  team-coloured boxes (from K-Means BGR cluster colours),
         bird's-eye minimap embedded in bottom-right corner.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from config import BALL_COLOR, TEAM_A_COLOR, TEAM_B_COLOR, TEXT_COLOR
from src.tracking.tracker import TrackingResult

# Fallback annotation colours (BGR)
_FALLBACK_COLORS_BGR: List[Tuple[int, int, int]] = [TEAM_A_COLOR, TEAM_B_COLOR]


def _bgr_to_sv_color(bgr: Tuple[int, int, int]) -> sv.Color:
    """Convert OpenCV BGR tuple to supervision Color (RGB order)."""
    b, g, r = bgr
    return sv.Color(r=r, g=g, b=b)


def _build_labels(detections: sv.Detections) -> List[str]:
    if detections.tracker_id is None:
        return ["?" for _ in range(len(detections))]
    return [f"#{int(tid)}" for tid in detections.tracker_id]


class FrameAnnotator:
    """
    Draws all per-frame visual overlays.

    annotate() is the main entry point — forward-compatible across weeks.
    """

    def annotate(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        team_assignments: Optional[Dict[int, int]] = None,
        team_colors_bgr: Optional[List[Tuple[int, int, int]]] = None,
        pitch_view: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return an annotated copy of frame with all overlays.

        Args:
            frame:            Original BGR frame.
            result:           TrackingResult from PlayerTracker.update().
            team_assignments: tracker_id → team_id (0 or 1).  None = Week 1 mode.
            team_colors_bgr:  BGR colour per team.  None = use fallback palette.
            pitch_view:       Full-res bird's-eye render.  None = no minimap.
        """
        out = frame.copy()
        colors = team_colors_bgr or _FALLBACK_COLORS_BGR

        out = self._draw_players(out, result, team_assignments, colors)
        out = self._draw_ball(out, result)
        out = self._draw_hud(out, result, team_assignments, colors)
        if pitch_view is not None:
            out = self._embed_minimap(out, pitch_view)
        return out

    # ── Players ───────────────────────────────────────────────────────────────

    def _draw_players(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        team_assignments: Optional[Dict[int, int]],
        team_colors_bgr: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        players = result.tracked_players
        if len(players) == 0:
            return frame

        if team_assignments and players.tracker_id is not None:
            # Draw each team's players separately with their team colour
            for team_id, color_bgr in enumerate(team_colors_bgr):
                mask = np.array(
                    [team_assignments.get(int(tid), 0) == team_id
                     for tid in players.tracker_id],
                    dtype=bool,
                )
                if not mask.any():
                    continue

                team_det = players[mask]
                sv_color = _bgr_to_sv_color(color_bgr)
                labels = _build_labels(team_det)

                box_ann = sv.BoxAnnotator(color=sv_color, thickness=2)
                lbl_ann = sv.LabelAnnotator(
                    text_scale=0.45,
                    text_thickness=1,
                    text_padding=4,
                    color=sv_color,
                    text_color=sv.Color.WHITE,
                )
                frame = sv.EllipseAnnotator(color=sv_color, thickness=2).annotate(
                    frame, team_det
                )
                frame = box_ann.annotate(frame, team_det)
                frame = lbl_ann.annotate(frame, team_det, labels)
        else:
            # Week 1 mode — single colour for all players
            labels = _build_labels(players)
            frame = sv.EllipseAnnotator(thickness=2).annotate(frame, players)
            frame = sv.BoxAnnotator(thickness=2).annotate(frame, players)
            frame = sv.LabelAnnotator(
                text_scale=0.45, text_thickness=1, text_padding=4
            ).annotate(frame, players, labels)

        return frame

    # ── Ball ──────────────────────────────────────────────────────────────────

    def _draw_ball(self, frame: np.ndarray, result: TrackingResult) -> np.ndarray:
        if result.ball is None or len(result.ball) == 0:
            return frame
        x1, y1, x2, y2 = result.ball.xyxy[0].astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 9, BALL_COLOR, -1)
        cv2.circle(frame, (cx, cy), 9, (0, 0, 0), 2)
        cv2.putText(frame, "Ball", (cx + 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, BALL_COLOR, 1, cv2.LINE_AA)
        return frame

    # ── HUD overlay ───────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        team_assignments: Optional[Dict[int, int]],
        team_colors_bgr: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        pad, lh = 14, 18
        n_teams_visible = len(team_colors_bgr) if team_assignments else 0
        n_lines = 3 + max(n_teams_visible, 0)
        box_w, box_h = 280, pad * 2 + lh * n_lines

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        lines = [
            f"Frame {result.frame_idx:04d}   t = {result.timestamp:.1f}s",
            f"Players tracked : {len(result.tracked_players)}",
            f"Ball            : {'detected' if result.ball is not None else 'not found'}",
        ]

        if team_assignments:
            from collections import Counter
            counts = Counter(team_assignments.values())
            for i, color_bgr in enumerate(team_colors_bgr):
                lines.append(f"  Team {chr(65+i)}        : {counts.get(i, 0)} players")

        for i, line in enumerate(lines):
            y = pad + 14 + i * lh
            col = TEXT_COLOR

            # Color-code team lines
            if team_assignments and i >= 3:
                team_idx = i - 3
                if team_idx < len(team_colors_bgr):
                    b, g, r = team_colors_bgr[team_idx]
                    col = (int(b), int(g), int(r))

            cv2.putText(frame, line, (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        return frame

    # ── Minimap embed ─────────────────────────────────────────────────────────

    def _embed_minimap(
        self,
        frame: np.ndarray,
        pitch_view: np.ndarray,
        scale: float = 0.28,
        margin: int = 12,
    ) -> np.ndarray:
        """Embed a scaled pitch minimap in the bottom-right corner."""
        h, w = frame.shape[:2]
        ph, pw = pitch_view.shape[:2]
        th = int(h * scale)
        tw = int(pw * th / ph)   # preserve aspect ratio

        mini = cv2.resize(pitch_view, (tw, th), interpolation=cv2.INTER_AREA)

        # Border
        cv2.rectangle(mini, (0, 0), (tw - 1, th - 1), (255, 255, 255), 2)

        y1 = h - th - margin
        x1 = w - tw - margin

        # Sanity bounds
        if y1 < 0 or x1 < 0:
            return frame

        roi = frame[y1:y1 + th, x1:x1 + tw]
        frame[y1:y1 + th, x1:x1 + tw] = cv2.addWeighted(roi, 0.15, mini, 0.85, 0)

        # "Bird's-eye" label
        cv2.putText(frame, "Bird's-eye", (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        return frame
