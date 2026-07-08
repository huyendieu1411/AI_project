import cv2
import pandas as pd

from config import VIDEO_PATH, TRACKING_DIR

# ==========================================================
# CONFIG
# ==========================================================

TRACKS_CSV = TRACKING_DIR / "tracks_boxmot.csv"

OUTPUT_VIDEO = TRACKING_DIR / "visualize_counter.mp4"

LINE_Y = 650

MIN_TRACK_LENGTH = 10


# ==========================================================
# Colors
# ==========================================================

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]

# ==========================================================
# Load tracking result
# ==========================================================

df = pd.read_csv(TRACKS_CSV)

# ----------------------------------------------------------
# Filter short tracks
# ----------------------------------------------------------

track_length = df.groupby("id")["frame"].nunique()

valid_ids = track_length[
    track_length >= MIN_TRACK_LENGTH
].index

df = df[df["id"].isin(valid_ids)]

# ==========================================================
# Build event list
# ==========================================================

count_events = {}

counted = set()

for track_id, track in df.groupby("id"):

    track = track.sort_values("frame")

    prev_y = None

    for _, row in track.iterrows():

        cy = row["cy"]

        if prev_y is not None:

            if (
                prev_y < LINE_Y
                and
                cy >= LINE_Y
                and
                track_id not in counted
            ):

                counted.add(track_id)

                count_events[int(track_id)] = int(row["frame"])

                break

        prev_y = cy

print("=" * 60)
print("Count events:", len(count_events))
print("=" * 60)

# ==========================================================
# Video
# ==========================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

assert cap.isOpened()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_id = 0

current_count = 0

already_counted = set()

# ==========================================================
# Draw video
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------------
    # Draw counting line
    # ----------------------------

    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (0, 255, 255),
        2
    )

    frame_data = df[df["frame"] == frame_id]

    # ----------------------------
    # Draw detections
    # ----------------------------

    for _, row in frame_data.iterrows():

        tid = int(row["id"])

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])
        conf = float(row["conf"])
	
        cx = int(row["cx"])
        cy = int(row["cy"])

        color = COLORS[tid % len(COLORS)]

        # Sau khi đếm thì đổi sang đỏ
        if tid in already_counted:
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            6,
            (0, 0, 255),
            -1
        )

        label = f"ID {tid} | {conf:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 8, y1),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,255),
            2
        )

        # --------------------------------
        # Count event
        # --------------------------------

        if (
            tid in count_events
            and
            count_events[tid] == frame_id
            and
            tid not in already_counted
        ):

            already_counted.add(tid)

            current_count += 1

            print(
                f"Frame {frame_id} | "
                f"ID {tid} | "
                f"COUNT = {current_count}"
            )

    # ----------------------------
    # Overlay
    # ----------------------------

    cv2.putText(
        frame,
        f"COUNT : {current_count}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        f"FRAME : {frame_id}",
        (40, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(frame)

    frame_id += 1

cap.release()

writer.release()

print("=" * 60)
print("Visualization completed")
print("Final Count :", current_count)
print("Saved :", OUTPUT_VIDEO)
print("=" * 60)
