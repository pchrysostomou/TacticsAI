"""
Week 3 — Tactical Metrics

Computes real-time tactical statistics from pitch-space positions:

  - **Team width / depth / compactness** (bounding box of all players)
  - **Pressing intensity** (avg distance to nearest opponent)
  - **Ball possession zone** (which third the ball sits in)
  - **Team centroid** (average position)
  - **Inter-player distance** (cohesion metric)

All metrics operate in pitch canvas pixels.  Where needed, scale by
(PITCH_REAL_W_M / PITCH_CANVAS_W) to convert to approximate metres.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import PITCH_CANVAS_H, PITCH_CANVAS_W, PITCH_REAL_H_M, PITCH_REAL_W_M

_PX_TO_M_X = PITCH_REAL_W_M / PITCH_CANVAS_W
_PX_TO_M_Y = PITCH_REAL_H_M / PITCH_CANVAS_H


def _px_to_m(px: float, axis: str = "x") -> float:
    return px * (_PX_TO_M_X if axis == "x" else _PX_TO_M_Y)


def compute_metrics(
    team_positions: Dict[int, List[Tuple[float, float]]],
    ball_position: Optional[Tuple[float, float]],
    pitch_w: int = PITCH_CANVAS_W,
    pitch_h: int = PITCH_CANVAS_H,
) -> Dict:
    """
    Compute a full tactical metrics dict for the current frame.

    Returns::
        {
          "teams": {
              team_id: {
                  "centroid": (px, py),
                  "width_m":  float,
                  "depth_m":  float,
                  "compactness_m": float,   # area in m²
                  "cohesion_m":   float,    # avg pairwise distance
              }
          },
          "pressing_intensity": {team_id: float},  # avg dist to nearest opp (m)
          "ball_zone":          str,                # "own_half" / "midfield" / "opposition"
          "space_control":      {team_id: float},   # fraction of pitch closer to team
        }
    """
    result: Dict = {"teams": {}, "pressing_intensity": {}, "ball_zone": None,
                    "space_control": {}}

    # ── Per-team metrics ──────────────────────────────────────────────────────
    for team_id, positions in team_positions.items():
        if len(positions) < 2:
            result["teams"][team_id] = {
                "centroid": positions[0] if positions else (0, 0),
                "width_m": 0, "depth_m": 0,
                "compactness_m": 0, "cohesion_m": 0,
            }
            continue

        pts = np.array(positions)
        cx, cy = pts.mean(axis=0)
        xs, ys = pts[:, 0], pts[:, 1]

        w_px = xs.max() - xs.min()
        d_px = ys.max() - ys.min()
        width_m     = _px_to_m(w_px, "x")
        depth_m     = _px_to_m(d_px, "y")
        compact_m2  = width_m * depth_m

        # Average pairwise distance (cohesion)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        n = len(pts)
        cohesion_px = dists[np.triu_indices(n, k=1)].mean() if n > 1 else 0.0
        cohesion_m  = _px_to_m(cohesion_px, "x")

        result["teams"][team_id] = {
            "centroid":      (float(cx), float(cy)),
            "width_m":       round(width_m, 1),
            "depth_m":       round(depth_m, 1),
            "compactness_m": round(compact_m2, 0),
            "cohesion_m":    round(cohesion_m, 1),
        }

    # ── Pressing intensity ────────────────────────────────────────────────────
    team_ids = list(team_positions.keys())
    if len(team_ids) >= 2:
        for i, t_a in enumerate(team_ids):
            t_b = team_ids[1 - i]
            pts_a = np.array(team_positions[t_a]) if team_positions[t_a] else None
            pts_b = np.array(team_positions[t_b]) if team_positions[t_b] else None
            if pts_a is None or pts_b is None or len(pts_a) == 0 or len(pts_b) == 0:
                result["pressing_intensity"][t_a] = 0.0
                continue
            # For each player in t_a, find nearest opponent
            diffs = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
            dist2 = (diffs ** 2).sum(axis=-1)
            min_dists_px = np.sqrt(dist2.min(axis=1))
            avg_press_px = min_dists_px.mean()
            result["pressing_intensity"][t_a] = round(_px_to_m(avg_press_px, "x"), 1)

    # ── Ball zone ─────────────────────────────────────────────────────────────
    if ball_position is not None:
        bx, by = ball_position
        third = pitch_w / 3
        if bx < third:
            result["ball_zone"] = "Own Third"
        elif bx < 2 * third:
            result["ball_zone"] = "Midfield"
        else:
            result["ball_zone"] = "Opposition Third"
    else:
        result["ball_zone"] = "—"

    # ── Voronoi-based space control (approximate) ─────────────────────────────
    all_teams = list(team_positions.keys())
    if len(all_teams) >= 2 and all(
        len(team_positions[t]) > 0 for t in all_teams[:2]
    ):
        t_a, t_b = all_teams[0], all_teams[1]
        pts_a = np.array(team_positions[t_a], dtype=np.float32)
        pts_b = np.array(team_positions[t_b], dtype=np.float32)

        # Sample a coarse grid and assign each point to the nearest team
        step = 20
        grid_x, grid_y = np.meshgrid(
            np.arange(0, pitch_w, step),
            np.arange(0, pitch_h, step),
        )
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        da = ((grid[:, np.newaxis, :] - pts_a[np.newaxis, :, :]) ** 2).sum(-1).min(-1)
        db = ((grid[:, np.newaxis, :] - pts_b[np.newaxis, :, :]) ** 2).sum(-1).min(-1)

        closer_a = (da < db).sum()
        total    = len(grid)
        result["space_control"][t_a] = round(closer_a / total * 100, 1)
        result["space_control"][t_b] = round((total - closer_a) / total * 100, 1)

    return result
