"""TacticsAI — Global configuration (Week 1 + Week 2)."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

# Create dirs on import
for _d in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    _d.mkdir(exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "yolov8x.pt"   # Base model; swap for "best.pt" after Colab training

# ── Detection ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5     # Min confidence to keep a detection
IOU_THRESHOLD = 0.45           # NMS IoU threshold

# COCO class IDs (relevant to football)
PERSON_CLASS_ID = 0            # Player / referee / photographer
BALL_CLASS_ID = 32             # Sports ball

# ── Tracking (ByteTrack) ──────────────────────────────────────────────────────
TRACK_ACTIVATION_THRESHOLD = 0.25   # Min confidence to activate a new track
LOST_TRACK_BUFFER = 30              # Frames to keep a lost track alive
MINIMUM_MATCHING_THRESHOLD = 0.8    # IoU threshold for track matching

# ── Video ─────────────────────────────────────────────────────────────────────
DEFAULT_FPS = 30

# ── Week 2: Team Classification ───────────────────────────────────────────────
CALIBRATION_FRAMES = 60      # Frames for jersey colour collection before fit
N_TEAMS = 2                  # k for K-Means (2 = Team A + Team B)

# ── Week 2: Pitch (Perspective Transform + Renderer) ──────────────────────────
PITCH_CANVAS_W = 1040        # Rendered pitch canvas width  (pixels)
PITCH_CANVAS_H = 680         # Rendered pitch canvas height (pixels)
# Real-world dimensions (FIFA standard)
PITCH_REAL_W_M = 105.0
PITCH_REAL_H_M = 68.0

# Default src_points — now auto-detected per-video in app.py / pipeline.
# These are a conservative fallback (exclude top ~180px stands/scoreboards).
# For the Tryolabs broadcast demo specifically:
#   far touchline y≈180, near touchline y≈710, full width x=0..1279
DEFAULT_SRC_POINTS = [
    [0,    180],
    [1279, 180],
    [1279, 710],
    [0,    710],
]

# How many pixels from the top of the frame to exclude from calibration
# (scoreboard, TV overlays, stands). Increase for heavily cropped broadcasts.
ADS_STRIP_HEIGHT = 130

# ── Colors (BGR — OpenCV convention) ─────────────────────────────────────────
# Team colors: used as fallback before K-Means is fitted (Week 2 derives
# actual colors from jersey HSV cluster centres).
TEAM_A_COLOR = (0, 80, 255)        # Orange-red (fallback)
TEAM_B_COLOR = (0, 210, 70)        # Green (fallback)
BALL_COLOR   = (0, 230, 255)       # Yellow
REFEREE_COLOR = (255, 220, 0)      # Cyan
TEXT_COLOR   = (255, 255, 255)     # White
HUD_BG_COLOR = (0, 0, 0)          # HUD background (with alpha)
