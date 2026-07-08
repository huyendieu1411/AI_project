from pathlib import Path
import pandas as pd

# ============================
# Paths
# ============================

INPUT = Path("runs/track/Video/tracks.txt")
OUTPUT = Path("results/tracking/tracks_boxmot.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ============================
# Read MOT txt
# ============================

df = pd.read_csv(
    INPUT,
    header=None,
    names=[
        "frame",
        "id",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "dummy"
    ]
)

# ============================
# Convert
# ============================

df["x1"] = df["x"]
df["y1"] = df["y"]

df["x2"] = df["x"] + df["w"]
df["y2"] = df["y"] + df["h"]

df["cx"] = (df["x1"] + df["x2"]) / 2
df["cy"] = df["y2"]

# ============================
# Keep same format as old CSV
# ============================

df = df[
    [
        "frame",
        "id",
        "conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy"
    ]
]

df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("Converted successfully")
print("=" * 60)
print(f"Rows : {len(df)}")
print(f"IDs  : {df['id'].nunique()}")
print(f"CSV  : {OUTPUT}")
print("=" * 60)
