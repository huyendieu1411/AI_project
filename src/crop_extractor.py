import cv2
import pandas as pd
from pathlib import Path

from config import *

# ==========================================================
# Create output folder
# ==========================================================

CROP_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load tracks
# ==========================================================

print("Loading tracks...")

df = pd.read_csv(TRACKING_CSV)

print(f"Detections : {len(df)}")

# ==========================================================
# Open video
# ==========================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

assert cap.isOpened(), "Cannot open video."

current_frame = 0

saved = 0

print("Extracting crops...")

# ==========================================================
# Main loop
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    detections = df[df["frame"] == current_frame]

    for _, row in detections.iterrows():

        if (not SAVE_ALL_CROPS) and (current_frame % DEBUG_EVERY_N_FRAMES != 0):
            continue

        x1 = max(0, int(row["x1"]))
        y1 = max(0, int(row["y1"]))
        x2 = min(frame.shape[1], int(row["x2"]))
        y2 = min(frame.shape[0], int(row["y2"]))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        filename = (
            f"frame_{current_frame:06d}"
            f"_id_{int(row['id'])}.jpg"
        )

        cv2.imwrite(
            str(CROP_DIR / filename),
            crop
        )

        saved += 1

    if current_frame % 200 == 0:
        print(
            f"Frame {current_frame}",
            end="\r"
        )

    current_frame += 1

cap.release()

print()

print("=" * 60)
print("Crop extraction completed")
print("=" * 60)
print(f"Frames processed : {current_frame}")
print(f"Crops saved      : {saved}")
print(f"Output folder    : {CROP_DIR}")
print("=" * 60)