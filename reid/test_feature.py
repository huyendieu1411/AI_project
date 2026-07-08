from pathlib import Path

from feature_extractor import FeatureExtractor


image = Path(
    "results/reid/crops/frame_000000_id_1.jpg"
)

extractor = FeatureExtractor()

feature = extractor.extract(str(image))

print()

print(feature.shape)

print()

print(feature[:20])