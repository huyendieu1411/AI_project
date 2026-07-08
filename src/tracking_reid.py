import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    VIDEO_PATH,
    TRACKING_DIR
)


# ==========================================================
# Config
# ==========================================================

TRACKER = "src/botsort_reid.yaml"

OUTPUT_VIDEO = TRACKING_DIR / "tracking_reid.mp4"
OUTPUT_CSV = TRACKING_DIR / "tracks_reid.csv"


# ==========================================================
# Load YOLO
# ==========================================================

print("=" * 60)
print("Loading YOLO model")
print("=" * 60)

model = YOLO(str(MODEL_PATH))


# ==========================================================
# Open video
# ==========================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

assert cap.isOpened(), "Cannot open video"


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# ==========================================================
# Video writer
# ==========================================================

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# ==========================================================
# Tracking
# ==========================================================

tracks = []

frame_id = 0


print("=" * 60)
print("Tracking with ReID started")
print("=" * 60)


results = model.track(
    source=str(VIDEO_PATH),
    tracker=TRACKER,
    persist=True,
    stream=True,
    save=False,
    conf=0.3,
    iou=0.5,
    verbose=False
)


for result in results:

    frame = result.orig_img

    annotated = result.plot()

    writer.write(annotated)


    if result.boxes.id is not None:

        ids = result.boxes.id.cpu().numpy().astype(int)

        boxes = result.boxes.xyxy.cpu().numpy()

        confs = result.boxes.conf.cpu().numpy()


        for track_id, box, conf in zip(ids, boxes, confs):

            x1, y1, x2, y2 = box


            tracks.append({

                "frame": frame_id,

                "id": int(track_id),

                "conf": float(conf),

                "x1": float(x1),

                "y1": float(y1),

                "x2": float(x2),

                "y2": float(y2),

                "cx": float((x1+x2)/2),

                "cy": float((y1+y2)/2)

            })


    frame_id += 1


    if frame_id % 500 == 0:
        print(f"Processed {frame_id} frames")


cap.release()
writer.release()


# ==========================================================
# Save CSV
# ==========================================================

df = pd.DataFrame(tracks)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ==========================================================
# Summary
# ==========================================================

print("=" * 60)
print("Tracking with ReID completed")
print("=" * 60)

print("Frames      :", frame_id)
print("Detections  :", len(df))

if len(df) > 0:
    print("Unique IDs  :", df["id"].nunique())

print()
print("Video :", OUTPUT_VIDEO)
print("CSV   :", OUTPUT_CSV)

print("=" * 60)