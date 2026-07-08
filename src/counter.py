import pandas as pd
from pathlib import Path

from config import TRACKING_DIR


# ==========================================================
# CONFIG
# ==========================================================

CSV_PATH = TRACKING_DIR / "tracks_boxmot.csv"


# Đường đếm
# Sau này có thể chỉnh
LINE_Y = 650


# Số frame tối thiểu để xem là người thật
MIN_TRACK_LENGTH = 10


# Hướng đi xuống:
# cy trước < line
# cy sau >= line
COUNT_DIRECTION = "down"


# ==========================================================
# Load tracking result
# ==========================================================

print("=" * 60)
print("Loading tracking CSV")
print("=" * 60)


df = pd.read_csv(CSV_PATH)


print("Total detections :", len(df))
print("Total IDs        :", df["id"].nunique())


# ==========================================================
# Filter short tracks
# ==========================================================

track_length = (
    df.groupby("id")["frame"]
    .nunique()
)


valid_ids = track_length[
    track_length >= MIN_TRACK_LENGTH
].index


df = df[
    df["id"].isin(valid_ids)
]


print()
print("=" * 60)
print("After filtering")
print("=" * 60)

print(
    "Valid IDs        :",
    df["id"].nunique()
)


# ==========================================================
# Count crossing line
# ==========================================================

counted_ids = set()

count_events = []


for track_id, track in df.groupby("id"):

    # sắp xếp theo thời gian
    track = track.sort_values("frame")


    previous_y = None


    for _, row in track.iterrows():

        current_y = row["cy"]


        if previous_y is not None:

            if (
                previous_y < LINE_Y
                and
                current_y >= LINE_Y
            ):

                if track_id not in counted_ids:

                    counted_ids.add(track_id)


                    count_events.append({

                        "id": int(track_id),

                        "frame": int(row["frame"]),

                        "previous_y": float(previous_y),

                        "current_y": float(current_y)

                    })


        previous_y = current_y



# ==========================================================
# Result
# ==========================================================

result = pd.DataFrame(count_events)


print()
print("=" * 60)
print("Counting completed")
print("=" * 60)


print(
    "People counted :",
    len(counted_ids)
)


if len(result) > 0:

    print("\nEvents:")

    print(result.to_string(index=False))


else:

    print("No crossing detected")


# Save result

OUTPUT = TRACKING_DIR / "count_result.csv"

result.to_csv(
    OUTPUT,
    index=False
)


print()
print("Saved:", OUTPUT)

print("=" * 60)
