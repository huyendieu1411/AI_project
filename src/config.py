from pathlib import Path

# ==========================================================
# Project paths
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_03_06.pt"
VIDEO_PATH = PROJECT_ROOT / "videos" / "Video.mp4"

RESULT_DIR = PROJECT_ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

TRACKING_DIR = RESULT_DIR / "tracking"
TRACKING_DIR.mkdir(exist_ok=True)

COUNTING_DIR = RESULT_DIR / "counting"
COUNTING_DIR.mkdir(exist_ok=True)

# ==========================================================
# Tracking
# ==========================================================
TRACKER = "botsort.yaml"

CONF = 0.30
IOU = 0.50

# ==========================================================
# Counting
# ==========================================================
LINE_Y = 650
MIN_TRACK_FRAMES = 10

# ==========================================================
# Output files
# ==========================================================
TRACKING_VIDEO = TRACKING_DIR / "tracking.mp4"
TRACKING_CSV = TRACKING_DIR / "tracks.csv"

COUNTING_VIDEO = COUNTING_DIR / "counting_result.mp4"

# ==========================================================
# ReID
# ==========================================================

REID_DIR = RESULT_DIR / "reid"
REID_DIR.mkdir(exist_ok=True)

CROP_DIR = REID_DIR / "crops"
CROP_DIR.mkdir(exist_ok=True)

STABLE_TRACKS_CSV = REID_DIR / "stable_tracks.csv"

REID_MODEL = "osnet_x0_25"

SIMILARITY_THRESHOLD = 0.85

MAX_GALLERY_AGE = 60

# ==========================================================
# Debug
# ==========================================================

SAVE_ALL_CROPS = False      # True = lưu toàn bộ crop
DEBUG_EVERY_N_FRAMES = 100  # Nếu SAVE_ALL_CROPS=False thì lưu 1 crop mỗi 100 frame
