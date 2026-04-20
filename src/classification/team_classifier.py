"""
Team classification via K-Means jersey color clustering.

Strategy:
  1. Collect HSV jersey-crop samples from the first N frames (calibration).
  2. Fit KMeans(k=2) on the accumulated samples.
  3. Classify each tracked player — cached per tracker_id for frame-to-frame
     stability (jersey colour never changes mid-match).
  4. Derive human-readable BGR display colours from cluster centres so that
     bounding boxes visually reflect the actual jersey colour.

Week 3 extension: bump n_teams=3 to separate referee colour automatically.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Fallback colours (BGR) used before the classifier is fitted
_FALLBACK_BGR: List[Tuple[int, int, int]] = [(0, 80, 255), (0, 210, 70)]


# ── Jersey colour extraction ──────────────────────────────────────────────────

def _extract_jersey_hsv(
    frame: np.ndarray,
    bbox: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Return the dominant HSV jersey colour from a player bounding box.

    Crops the torso region (25 %–65 % of bbox height) to capture the shirt
    while excluding the head, shorts and boots.  Pixels that are very dark
    (shadow) or very bright / low-saturation (pitch lines, sky) are discarded.
    A fast 2-cluster KMeans separates jersey pixels from background.

    Returns:
        1-D float32 array [H, S, V] or None if the crop is too small.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h = max(y2 - y1, 1)

    # Torso crop
    top = y1 + int(h * 0.25)
    bot = y1 + int(h * 0.65)
    crop = frame[top:bot, x1:x2]

    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Reject shadows and pitch-line whites
    v, s = pixels[:, 2], pixels[:, 1]
    mask = (v > 35) & (v < 235) & (s > 20)
    filtered = pixels[mask]

    if len(filtered) < 10:
        return None

    # Fast 2-cluster KMeans → dominant jersey colour vs background
    try:
        km = KMeans(n_clusters=2, n_init=1, max_iter=20, random_state=0)
        km.fit(filtered)
        counts = np.bincount(km.labels_)
        return km.cluster_centers_[int(np.argmax(counts))]
    except Exception:
        return filtered.mean(axis=0)


# ── Classifier ────────────────────────────────────────────────────────────────

class TeamClassifier:
    """
    Multi-frame K-Means team classifier.

    Lifecycle::
        clf = TeamClassifier()

        # Phase 1 — calibration (first N frames)
        for bbox in player_bboxes:
            clf.collect(frame, bbox)
        clf.fit()          # call once after calibration frames

        # Phase 2 — per-frame classification
        team_id = clf.classify(frame, bbox, tracker_id)  # 0 or 1
    """

    def __init__(self, n_teams: int = 2) -> None:
        self.n_teams = n_teams
        self._kmeans: Optional[KMeans] = None
        self.is_fitted = False

        self._samples: List[np.ndarray] = []          # calibration buffer
        self._cache: Dict[int, int] = {}              # tracker_id → team_id
        self._colors_bgr: List[Tuple[int, int, int]] = list(_FALLBACK_BGR)

    # ── Calibration ───────────────────────────────────────────────────────────

    def collect(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        """Add one jersey HSV sample to the calibration buffer."""
        color = _extract_jersey_hsv(frame, bbox)
        if color is not None:
            self._samples.append(color)

    def fit(self) -> bool:
        """
        Fit KMeans on collected jersey samples.

        Returns True on success, False if too few samples were collected.
        """
        if len(self._samples) < self.n_teams * 12:
            return False

        X = np.array(self._samples, dtype=np.float32)
        self._kmeans = KMeans(
            n_clusters=self.n_teams,
            n_init=15,
            max_iter=300,
            random_state=42,
        )
        self._kmeans.fit(X)
        self.is_fitted = True

        # Build display colours from cluster centres (HSV → BGR, boosted)
        self._colors_bgr = []
        for center in self._kmeans.cluster_centers_:
            hsv_u8 = np.uint8([[center]])
            bgr = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)[0][0]
            # Boost saturation + ensure minimum brightness for visibility
            hsv_b = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0].astype(int)
            hsv_b[1] = min(255, int(hsv_b[1] * 1.5))
            hsv_b[2] = max(130, int(hsv_b[2]))
            hsv_b = np.clip(hsv_b, 0, 255).astype(np.uint8)
            bgr_out = cv2.cvtColor(np.uint8([[hsv_b]]), cv2.COLOR_HSV2BGR)[0][0]
            self._colors_bgr.append(tuple(int(c) for c in bgr_out))

        return True

    # ── Per-frame classification ───────────────────────────────────────────────

    def classify(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        tracker_id: int,
    ) -> int:
        """
        Return team_id (0 or 1) for a tracked player.

        Results are cached by tracker_id — once classified a player keeps the
        same team label for the duration of the video.
        """
        if not self.is_fitted:
            return 0

        if tracker_id in self._cache:
            return self._cache[tracker_id]

        color = _extract_jersey_hsv(frame, bbox)
        if color is None:
            return 0

        team_id = int(self._kmeans.predict(color.reshape(1, -1))[0])
        self._cache[tracker_id] = team_id
        return team_id

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def team_colors_bgr(self) -> List[Tuple[int, int, int]]:
        """BGR display colours for each team (derived from jersey colours)."""
        return self._colors_bgr

    def team_counts(self, assignments: Dict[int, int]) -> Dict[int, int]:
        """Count players per team from tracker_id → team_id dict."""
        counts: Dict[int, int] = {i: 0 for i in range(self.n_teams)}
        for team_id in assignments.values():
            counts[team_id] = counts.get(team_id, 0) + 1
        return counts

    def reset(self) -> None:
        """Full reset — call between different video sources."""
        self._kmeans = None
        self.is_fitted = False
        self._samples.clear()
        self._cache.clear()
        self._colors_bgr = list(_FALLBACK_BGR)
