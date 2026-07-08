import cv2
import pandas as pd
from ultralytics import YOLO

from config import *

# ==========================================================
# Initialize
# ==========================================================

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    str(TRACKING_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

tracks = []

frame_idx = 0

print("=" * 60)
print("Tracking started")
print("=" * 60)

# ==========================================================
# Main loop
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        tracker=TRACKER,
        persist=True,
        conf=CONF,
        iou=IOU,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    writer.write(annotated)

    if result.boxes.id is not None:

        ids = result.boxes.id.cpu().numpy().astype(int)

        boxes = result.boxes.xyxy.cpu().numpy()

        confs = result.boxes.conf.cpu().numpy()

        for tid, box, conf in zip(ids, boxes, confs):

            x1, y1, x2, y2 = box

            tracks.append({

                "frame": frame_idx,

                "id": tid,

                "conf": float(conf),

                "x1": float(x1),

                "y1": float(y1),

                "x2": float(x2),

                "y2": float(y2),

                "cx": float((x1 + x2) / 2),

                "cy": float((y1 + y2) / 2)

            })

    frame_idx += 1

    if frame_idx % 200 == 0:

        print(
            f"Processed {frame_idx}/{total_frames} frames",
            end="\r"
        )

# ==========================================================
# Save CSV
# ==========================================================

df = pd.DataFrame(tracks)

df.to_csv(TRACKING_CSV, index=False)

# ==========================================================
# Release
# ==========================================================

cap.release()

writer.release()

# ==========================================================
# Summary
# ==========================================================

print()
print("=" * 60)
print("Tracking completed")
print("=" * 60)

print(f"Frames      : {frame_idx}")
print(f"Detections  : {len(df)}")
print(f"Unique IDs  : {df['id'].nunique()}")

print()

print(f"Tracking Video : {TRACKING_VIDEO}")
print(f"Tracking CSV   : {TRACKING_CSV}")

print("=" * 60)
