"""
Pitch auto-calibration from video frame.

Detects pitch boundary from:
  1. White pitch-line detection (Hough) → find touchlines + halfway line
  2. Green grass segmentation → pitch convex hull corners
  3. Combining both for robust corner estimation

For broadcast side-view cameras: finds far touchline (y_min),
near touchline (y_max), and horizontal extent (x bounds).

Returns src_points in [TL, TR, BR, BL] order suitable for
PerspectiveTransformer.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def _detect_pitch_lines(
    frame: np.ndarray,
    min_y: int = 120,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Detect key pitch lines using Hough transform on white pixels.

    Uses bi-modal gap detection to separate the far touchline cluster
    from the near touchline cluster, handling broadcast cameras where
    both lines may be visible at very different Y positions.

    Returns:
        (far_touchline_y, near_touchline_y, halfway_line_x)
        Any may be None if not detected.
    """
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White line mask: high V, low S
    white = cv2.inRange(hsv, (0, 0, 185), (180, 45, 255))
    white[:min_y, :] = 0   # exclude stands/overlays at top

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)

    lines = cv2.HoughLinesP(
        white, 1, np.pi / 180,
        threshold=60, minLineLength=80, maxLineGap=40,
    )

    if lines is None:
        return None, None, None

    h_ys: List[int] = []
    v_xs: List[int] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-6))
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        if angle < 15 and mid_y > min_y:
            h_ys.append(mid_y)
        elif angle > 70:
            v_xs.append(mid_x)

    if len(h_ys) < 2:
        return None, None, (int(np.median(v_xs)) if v_xs else None)

    h_arr = np.array(sorted(h_ys))

    # Bi-modal split: find the largest gap between consecutive sorted Y values.
    # This separates the far-touchline cluster from the near-touchline cluster.
    gaps    = np.diff(h_arr)
    split_i = int(np.argmax(gaps))        # index of the largest inter-cluster gap

    top_cluster    = h_arr[:split_i + 1]  # small Y  → far (top of image)
    bottom_cluster = h_arr[split_i + 1:]  # large Y  → near (bottom of image)

    far_y = int(np.median(top_cluster)) if len(top_cluster) > 0 else None

    if len(bottom_cluster) > 0:
        near_y_cand = int(np.median(bottom_cluster))
        # Require the near line to be meaningfully below the far line
        near_y = near_y_cand if (far_y is None or near_y_cand > far_y + 100) else None
    else:
        # Fallback: use 90th percentile of entire set as near_y
        near_y = int(np.percentile(h_arr, 90)) if len(h_arr) >= 3 else None

    half_x = int(np.median(v_xs)) if len(v_xs) > 1 else None

    return far_y, near_y, half_x


def _detect_grass_corners(
    frame: np.ndarray,
    y_min: int = 120,
) -> Optional[List[List[int]]]:
    """
    Segment grass area and return convex-hull extreme corners [TL,TR,BR,BL].
    Returns None if grass coverage < 20%.
    """
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (25, 50, 40),  (85, 255, 255)),
        cv2.inRange(hsv, (25, 25, 60),  (90, 120, 220)),
    )
    mask[:y_min, :] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    area = cv2.contourArea(hull)
    if area < 0.20 * W * H:
        return None

    pts = hull.reshape(-1, 2)
    tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    tr = pts[np.argmax(pts[:, 0] - pts[:, 1])]
    br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]

    def clamp(p: np.ndarray) -> List[int]:
        return [int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1))]

    return [clamp(tl), clamp(tr), clamp(br), clamp(bl)]


def auto_detect_pitch_corners(
    frame: np.ndarray,
    ads_strip_height: int = 120,
) -> List[List[int]]:
    """
    Auto-detect the 4 pitch corners (image coords) from a video frame.

    Strategy:
      1. Find far/near touchlines via Hough white-line detection.
      2. Find grass convex hull for the X extent.
      3. Combine: use touchline Y values + grass X extent.
      4. Fallback to grass corners or frame boundary.

    Args:
        frame:            BGR video frame.
        ads_strip_height: Pixels from top to exclude (stands/scoreboards).

    Returns:
        [[TL_x, TL_y], [TR_x, TR_y], [BR_x, BR_y], [BL_x, BL_y]]
    """
    H, W = frame.shape[:2]

    far_y, near_y, half_x = _detect_pitch_lines(frame, min_y=ads_strip_height)
    grass  = _detect_grass_corners(frame, y_min=ads_strip_height)

    # Determine X extent ─────────────────────────────────────────────────────
    if grass:
        left_x  = min(grass[0][0], grass[3][0])   # TL.x, BL.x
        right_x = max(grass[1][0], grass[2][0])   # TR.x, BR.x
    else:
        left_x, right_x = 0, W - 1

    left_x  = max(0, left_x)
    right_x = min(W - 1, right_x)

    # Determine Y extent ─────────────────────────────────────────────────────
    if far_y is not None and near_y is not None and near_y > far_y + 50:
        top_y    = far_y
        bottom_y = near_y
    elif grass:
        top_y    = min(grass[0][1], grass[1][1])   # TL.y / TR.y
        bottom_y = max(grass[2][1], grass[3][1])   # BR.y / BL.y
    else:
        top_y    = ads_strip_height
        bottom_y = H - 10

    # Clamp ──────────────────────────────────────────────────────────────────
    top_y    = max(0, top_y)
    bottom_y = min(H - 1, bottom_y)

    return [
        [left_x,  top_y],     # TL
        [right_x, top_y],     # TR
        [right_x, bottom_y],  # BR
        [left_x,  bottom_y],  # BL
    ]


def validate_transform(
    frame: np.ndarray,
    src_points: List[List[float]],
    dst_width: int = 1040,
    dst_height: int = 680,
) -> np.ndarray:
    """
    Apply perspective transform and return warped frame.

    Use as visual sanity check: if the grass area fills the entire
    canvas, the calibration is correct.
    """
    src = np.float32(src_points)
    dst = np.float32([
        [0,          0],
        [dst_width,  0],
        [dst_width,  dst_height],
        [0,          dst_height],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (dst_width, dst_height))


def draw_corners_on_frame(
    frame: np.ndarray,
    corners: List[List[int]],
    color_bgr: Tuple[int, int, int] = (75, 123, 245),
) -> np.ndarray:
    """Overlay corner markers and quadrilateral on a frame copy."""
    out = frame.copy()
    labels = ["TL", "TR", "BR", "BL"]
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=color_bgr, thickness=3)
    for i, (pt, lbl) in enumerate(zip(corners, labels)):
        cv2.circle(out, (pt[0], pt[1]), 10, color_bgr, -1)
        cv2.putText(out, lbl, (pt[0] + 12, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2, cv2.LINE_AA)
    return out
