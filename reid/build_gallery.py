from collections import defaultdict

import cv2
import pandas as pd
from tqdm import tqdm

from feature_extractor import FeatureExtractor


class GalleryBuilder:

    def __init__(
        self,
        video_path,
        csv_path,
        sample_every=5
    ):

        self.video_path = video_path
        self.csv_path = csv_path
        self.sample_every = sample_every

        self.extractor = FeatureExtractor()

    def build(self):

        df = pd.read_csv(self.csv_path)

        gallery = defaultdict(list)

        cap = cv2.VideoCapture(self.video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("=" * 60)
        print("Building Gallery")
        print("=" * 60)

        for frame_idx in tqdm(range(total_frames)):

            ok, frame = cap.read()

            if not ok:
                break

            if frame_idx % self.sample_every != 0:
                continue

            rows = df[df.frame == frame_idx]

            for _, row in rows.iterrows():

                track_id = int(row.id)

                x1 = int(row.x1)
                y1 = int(row.y1)
                x2 = int(row.x2)
                y2 = int(row.y2)

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                feature = self.extractor.extract(crop)

                gallery[track_id].append({

                    "frame": frame_idx,

                    "bbox": [x1, y1, x2, y2],

                    "feature": feature

                })

        cap.release()

        print()

        print("Gallery IDs :", len(gallery))

        return gallery