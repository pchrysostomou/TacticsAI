"""
Perspective transform: camera image coordinates → top-down pitch coordinates.

Maps 4 user-defined pitch corner points (in image space) to a rectangle
representing the standard football pitch canvas.

The resulting pitch-space coordinates are passed to PitchRenderer so that
player circles are positioned accurately on the 2D pitch diagram.

Usage::
    # Define visible pitch corners in image coordinates:
    # [top-left, top-right, bottom-right, bottom-left]
    src = [[120, 80], [1160, 80], [1260, 660], [20, 660]]

    transformer = PerspectiveTransformer(src_points=src)

    # Per player:
    feet_x, feet_y = transformer.get_feet(bbox)
    pitch_x, pitch_y = transformer.transform_point(feet_x, feet_y)
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class PerspectiveTransformer:
    """
    Wraps cv2.getPerspectiveTransform for bidirectional camera ↔ pitch mapping.

    Args:
        src_points:  4 corner points in image coordinates
                     [top-left, top-right, bottom-right, bottom-left].
                     Pass None to use identity mode (no transform).
        dst_width:   Width of the pitch canvas output (pixels).
        dst_height:  Height of the pitch canvas output (pixels).
    """

    def __init__(
        self,
        src_points: Optional[List[List[float]]],
        dst_width: int = 1040,
        dst_height: int = 680,
    ) -> None:
        self.dst_w = dst_width
        self.dst_h = dst_height
        self.is_calibrated = src_points is not None
        self._M: Optional[np.ndarray] = None
        self._M_inv: Optional[np.ndarray] = None

        if src_points is not None:
            src = np.float32(src_points)
            dst = np.float32([
                [0,          0],
                [dst_width,  0],
                [dst_width,  dst_height],
                [0,          dst_height],
            ])
            self._M     = cv2.getPerspectiveTransform(src, dst)
            self._M_inv = cv2.getPerspectiveTransform(dst, src)

    # ── Forward transform (image → pitch) ─────────────────────────────────────

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Map a single image-space point to pitch canvas coordinates."""
        if self._M is None:
            # Identity: scale image coords to canvas size is not ideal but safe
            return x, y
        pt = np.float32([[[x, y]]])
        out = cv2.perspectiveTransform(pt, self._M)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Map (N, 2) image-space points to pitch canvas coordinates.

        Returns:
            (N, 2) float32 array in pitch canvas space.
        """
        if self._M is None or len(points) == 0:
            return points.astype(np.float32)
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self._M)
        return out.reshape(-1, 2)

    # ── Inverse transform (pitch → image) ─────────────────────────────────────

    def inverse_transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Map a pitch canvas point back to image-space."""
        if self._M_inv is None:
            return x, y
        pt = np.float32([[[x, y]]])
        out = cv2.perspectiveTransform(pt, self._M_inv)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_feet(bbox: np.ndarray) -> Tuple[float, float]:
        """
        Return the bottom-centre of a bounding box.

        Using the feet position (rather than centroid) gives a more accurate
        ground-plane location for the perspective transform.
        """
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, float(y2)

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Warp an entire frame to bird's-eye view (useful for debugging)."""
        if self._M is None:
            return frame
        return cv2.warpPerspective(frame, self._M, (self.dst_w, self.dst_h))
