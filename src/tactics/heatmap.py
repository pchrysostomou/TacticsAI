"""
Week 3 — Player Heatmaps

Accumulates foot-position history per team and renders Gaussian-kernel
density heatmaps on the pitch canvas.

The heatmap is computed on-demand from the raw position buffer, then
alpha-blended over the base pitch image.

Usage::
    hm = HeatmapEngine(pitch_w=1040, pitch_h=680)

    # Call every frame (or every N frames)
    hm.add_positions(team_positions)   # {team_id: [(px,py),...]}

    # Render current cumulative heatmap for one team (or both)
    img = hm.render(team_id=0, base_image=pitch_base)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Kernel bandwidth — higher = smoother, more spread heatmap (pixels)
_SIGMA = 30

# Alpha for overlay blending (0=base only, 1=heatmap only)
_HEAT_ALPHA = 0.55

# Colormap: COLORMAP_JET, INFERNO, HOT, MAGMA, PLASMA, TURBO
_CMAP = cv2.COLORMAP_INFERNO


class HeatmapEngine:
    """
    Accumulates player positions and renders Gaussian-blurred heatmaps.

    Args:
        pitch_w:   Pitch canvas width  (pixels).
        pitch_h:   Pitch canvas height (pixels).
        max_pts:   Rolling cap on stored positions (per team).
                   Older positions are dropped to bound memory.
    """

    def __init__(
        self,
        pitch_w: int = 1040,
        pitch_h: int = 680,
        max_pts: int = 50_000,
    ) -> None:
        self.w = pitch_w
        self.h = pitch_h
        self._max = max_pts
        # {team_id: list of (px, py)}
        self._history: Dict[int, List[Tuple[float, float]]] = {}

    # ── Data accumulation ─────────────────────────────────────────────────────

    def add_positions(
        self, team_positions: Dict[int, List[Tuple[float, float]]]
    ) -> None:
        """Append current-frame foot positions for each team."""
        for team_id, positions in team_positions.items():
            buf = self._history.setdefault(team_id, [])
            buf.extend(positions)
            if len(buf) > self._max:
                del buf[: len(buf) - self._max]

    def n_samples(self, team_id: int) -> int:
        """Number of stored position samples for a team."""
        return len(self._history.get(team_id, []))

    def reset(self, team_id: Optional[int] = None) -> None:
        """Clear history. If team_id is None, clear all teams."""
        if team_id is None:
            self._history.clear()
        else:
            self._history.pop(team_id, None)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        team_id: int,
        base_image: np.ndarray,
        alpha: float = _HEAT_ALPHA,
    ) -> np.ndarray:
        """
        Return the base pitch with a heatmap for team_id blended on top.

        Args:
            team_id:    Which team's heatmap to draw.
            base_image: BGR pitch canvas (H×W×3).  Not modified in-place.
            alpha:      Heatmap opacity (0–1).

        Returns:
            New BGR image with heatmap overlay.
        """
        pts = self._history.get(team_id, [])
        if len(pts) < 5:
            return base_image.copy()

        density = self._build_density(pts)
        colored = self._density_to_color(density)

        # Blend: areas of zero density stay transparent
        mask = (density > 0.01).astype(np.float32)[:, :, np.newaxis]
        out = base_image.copy().astype(np.float32)
        colored_f = colored.astype(np.float32)
        out = out * (1 - alpha * mask) + colored_f * (alpha * mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    def render_both(
        self,
        base_image: np.ndarray,
        team_colors_bgr: Optional[List[Tuple[int, int, int]]] = None,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Render combined heatmap for BOTH teams simultaneously using
        per-team single-channel density maps.

        Team A → hue derived from team color (red channel dominant).
        Team B → hue derived from team color (blue channel dominant).
        """
        out = base_image.copy().astype(np.float32)
        default_colors = [(0, 80, 255), (0, 210, 70)]
        colors = team_colors_bgr or default_colors

        for team_id, color_bgr in enumerate(colors):
            pts = self._history.get(team_id, [])
            if len(pts) < 5:
                continue
            density = self._build_density(pts)
            mask = density[:, :, np.newaxis]
            # Color: use the team BGRcolor tinted by density
            b, g, r = color_bgr
            layer = np.zeros_like(out)
            layer[:, :, 0] = b * mask[:, :, 0]
            layer[:, :, 1] = g * mask[:, :, 0]
            layer[:, :, 2] = r * mask[:, :, 0]
            m = mask
            out = out * (1 - alpha * m) + layer * (alpha * m)

        return np.clip(out, 0, 255).astype(np.uint8)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_density(self, pts: List[Tuple[float, float]]) -> np.ndarray:
        """
        Build a normalised Gaussian-blurred density map from raw points.
        Returns float32 array in [0, 1] of shape (H, W).
        """
        canvas = np.zeros((self.h, self.w), dtype=np.float32)
        for px, py in pts:
            x = int(np.clip(px, 0, self.w - 1))
            y = int(np.clip(py, 0, self.h - 1))
            canvas[y, x] += 1.0

        # Gaussian blur for density spread
        k = int(_SIGMA * 3) | 1   # odd kernel
        blurred = cv2.GaussianBlur(canvas, (k, k), sigmaX=_SIGMA, sigmaY=_SIGMA)

        # Normalise to [0, 1]
        max_val = blurred.max()
        if max_val > 0:
            blurred /= max_val
        return blurred

    @staticmethod
    def _density_to_color(density: np.ndarray) -> np.ndarray:
        """Convert normalised density map to BGR jet colormap image."""
        uint8 = (density * 255).astype(np.uint8)
        return cv2.applyColorMap(uint8, _CMAP)
