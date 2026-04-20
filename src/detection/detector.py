"""
YOLOv8 player and ball detector.

Wraps Ultralytics YOLO to:
  - Filter COCO class 0 (person) → players / referee
  - Filter COCO class 32 (sports ball) → ball
  - Return strongly-typed DetectionResult per frame
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import (
    BALL_CLASS_ID,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    PERSON_CLASS_ID,
)


@dataclass
class DetectionResult:
    """Raw YOLO detections for a single frame, split by object class."""

    players: sv.Detections          # All detected persons (class 0)
    ball: Optional[sv.Detections]   # Highest-confidence ball, or None
    raw: sv.Detections              # All detections (unfiltered)


class PlayerDetector:
    """
    Thin YOLOv8 wrapper for football-specific detection.

    Usage::
        detector = PlayerDetector("yolov8x.pt", device="cuda")
        result = detector.detect(frame)   # numpy BGR frame
        print(result.players.xyxy)        # bounding boxes
    """

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        device: str = "cuda",
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        print(f"[Detector] model={model_path}  device={device}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on one BGR frame.

        Args:
            frame: H×W×3 numpy array (BGR, from OpenCV).

        Returns:
            DetectionResult with .players and .ball populated.
        """
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )[0]

        all_det = sv.Detections.from_ultralytics(results)

        # ── Split by class ────────────────────────────────────────────────────
        if len(all_det) == 0:
            empty = sv.Detections.empty()
            return DetectionResult(players=empty, ball=None, raw=empty)

        player_mask = all_det.class_id == PERSON_CLASS_ID
        ball_mask = all_det.class_id == BALL_CLASS_ID

        players = all_det[player_mask]
        ball_candidates = all_det[ball_mask]

        # Keep only the highest-confidence ball if multiple detected
        ball: Optional[sv.Detections] = None
        if len(ball_candidates) > 0:
            best = int(np.argmax(ball_candidates.confidence))
            ball = ball_candidates[[best]]

        return DetectionResult(players=players, ball=ball, raw=all_det)
