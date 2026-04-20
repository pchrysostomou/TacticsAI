"""
Feature 6: Appearance-based Player Re-Identification.

Maintains a gallery of jersey-colour HSV histograms per tracker_id.
When ByteTrack loses a player and assigns a new ID, this module checks
appearance similarity against recently-lost IDs and reassigns the old ID
if the cosine similarity exceeds a threshold.

This is a lightweight solution that complements (not replaces) ByteTrack
motion-based re-ID. It is particularly useful when a player briefly exits
the frame (e.g. at touchlines).

Usage::
    reid = AppearanceReID()
    # Each frame:
    id_map = reid.update(frame, tracked_detections, team_assignments)
    # id_map: {new_tracker_id: canonical_tracker_id}
    # Apply remapping in pipeline before downstream analysis.
"""

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

# How many frames to keep appearance in the "lost" gallery
_GALLERY_FRAMES = 90

# Cosine similarity threshold to consider two IDs the same player
_SIM_THRESHOLD = 0.88

# HSV histogram parameters
_H_BINS, _S_BINS = 18, 16
_HIST_RANGE = [0, 180, 0, 256]


def _jersey_hist(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """Extract normalised HSV histogram from the torso crop of a player."""
    x1, y1, x2, y2 = map(int, bbox)
    h = max(y2 - y1, 1)
    crop = frame[y1 + int(h * 0.2): y1 + int(h * 0.6), x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [_H_BINS, _S_BINS],
        _HIST_RANGE,
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class AppearanceReID:
    """
    Lightweight appearance gallery for player re-identification.

    The gallery stores the last N HSV histograms for each active tracker_id
    (rolling average). When a player is not seen for a frame, their descriptor
    moves to a 'lost' buffer. On first appearance of a new ID, it is compared
    against the lost buffer for potential reassignment.
    """

    def __init__(
        self,
        sim_threshold: float = _SIM_THRESHOLD,
        gallery_frames: int = _GALLERY_FRAMES,
    ) -> None:
        self.sim_threshold = sim_threshold
        self.gallery_frames = gallery_frames

        # active_gallery: tracker_id → rolling mean histogram
        self._active: Dict[int, np.ndarray] = {}
        # lost_gallery: {old_tracker_id: (histogram, frames_since_lost)}
        self._lost: Dict[int, Tuple[np.ndarray, int]] = {}
        # canonical map: new_id → old_id (remapping)
        self._canonical: Dict[int, int] = {}

    # ── Per-frame update ─────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> Dict[int, int]:
        """
        Update gallery and return {tracker_id: canonical_id} mapping.

        Call this BEFORE using tracker_ids downstream so that any
        reassigned IDs are consistent.
        """
        if detections.tracker_id is None or len(detections) == 0:
            self._age_lost()
            return dict(self._canonical)

        visible_ids = set(int(t) for t in detections.tracker_id)

        # Age lost gallery (+1 frame each tick, evict old entries)
        self._age_lost()

        # Move IDs that disappeared into lost gallery
        for tid in list(self._active.keys()):
            if tid not in visible_ids:
                self._lost[tid] = (self._active.pop(tid), 0)

        # Update active gallery + check new IDs against lost gallery
        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            hist = _jersey_hist(frame, detections.xyxy[i])
            if hist is None:
                continue

            if tid in self._active:
                # Rolling mean (EMA)
                self._active[tid] = 0.85 * self._active[tid] + 0.15 * hist
            else:
                # New tracker_id — check if it matches a lost player
                best_match, best_sim = None, 0.0
                for old_id, (old_hist, _) in self._lost.items():
                    sim = _cosine_sim(hist, old_hist)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = old_id

                if best_match is not None and best_sim >= self.sim_threshold:
                    # Reassign: tid is actually best_match
                    self._canonical[tid] = best_match
                    old_hist, _ = self._lost.pop(best_match)
                    self._active[tid] = 0.5 * old_hist + 0.5 * hist
                else:
                    self._active[tid] = hist

        return dict(self._canonical)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _age_lost(self) -> None:
        to_remove = []
        for tid in self._lost:
            hist, age = self._lost[tid]
            if age + 1 >= self.gallery_frames:
                to_remove.append(tid)
            else:
                self._lost[tid] = (hist, age + 1)
        for tid in to_remove:
            del self._lost[tid]

        # Cap _canonical at 2000 entries — evict oldest (lowest) IDs
        if len(self._canonical) > 2000:
            excess = sorted(self._canonical.keys())[: len(self._canonical) - 2000]
            for k in excess:
                del self._canonical[k]

    def canonical_id(self, tracker_id: int) -> int:
        """Return canonical ID (may differ from tracker_id after re-ID)."""
        return self._canonical.get(tracker_id, tracker_id)

    def reset(self) -> None:
        self._active.clear()
        self._lost.clear()
        self._canonical.clear()
