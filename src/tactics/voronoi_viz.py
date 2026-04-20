"""
Feature 2: Voronoi Diagram Overlay.

For each pixel of the pitch canvas, assigns it to the nearest player
and colors it with that player's team color at reduced opacity.

Uses a fast distance-transform approximation (rasterising player
centroids into a label image, then using connected components).
For pitches with 20+ players, the pixel-wise scipy Voronoi approach
is accurate but slow — we use cv2.distanceTransform + argmin for speed.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def render_voronoi(
    canvas: np.ndarray,
    pitch_positions_by_id: Dict[int, Tuple[float, float]],
    team_by_id: Dict[int, int],
    team_colors_bgr: List[Tuple[int, int, int]],
    alpha: float = 0.28,
    n_teams: int = 2,
) -> np.ndarray:
    """
    Render approximate Voronoi control zones on a copy of canvas.

    Strategy:
      1. Build a (H, W, N_players) distance map.
      2. argmin over player axis → label map (which player owns pixel).
      3. Map label → team color → alpha-blend over canvas.

    Args:
        canvas:                  BGR pitch image (base).
        pitch_positions_by_id:   tracker_id → (px, py) on canvas.
        team_by_id:              tracker_id → team_id.
        team_colors_bgr:         BGR color per team.
        alpha:                   Voronoi layer opacity (0 = invisible).
        n_teams:                 Number of teams.

    Returns:
        New BGR image with Voronoi overlay.
    """
    if not pitch_positions_by_id:
        return canvas.copy()

    H, W = canvas.shape[:2]
    ids   = list(pitch_positions_by_id.keys())
    pts   = np.array([pitch_positions_by_id[i] for i in ids], dtype=np.float32)
    teams = np.array([team_by_id.get(i, 0)     for i in ids], dtype=np.int32)

    # Fast pixel-wise nearest-player via broadcasting on a coarse grid
    # then upscale → much faster than per-pixel iteration
    SCALE = 4   # compute at 1/4 resolution, upsample
    hw, ww = H // SCALE, W // SCALE

    gy, gx = np.mgrid[0:hw, 0:ww]
    grid   = np.stack([gx * SCALE, gy * SCALE], axis=-1).reshape(-1, 2).astype(np.float32)

    # (N_grid, N_players) distances squared
    diff  = grid[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    label = np.argmin(dist2, axis=1).reshape(hw, ww)  # player index

    # Map label → team color
    color_map = np.zeros((hw, ww, 3), dtype=np.uint8)
    for p_idx, (pid, team_id) in enumerate(zip(ids, teams)):
        mask = label == p_idx
        c = team_colors_bgr[team_id % len(team_colors_bgr)]
        color_map[mask] = (c[0], c[1], c[2])

    # Upscale to full canvas size
    color_full = cv2.resize(color_map, (W, H), interpolation=cv2.INTER_NEAREST)

    # Blend
    out = canvas.copy().astype(np.float32)
    out = out * (1 - alpha) + color_full.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
