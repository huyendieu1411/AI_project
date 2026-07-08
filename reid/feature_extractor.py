from pathlib import Path

import cv2
import numpy as np
import torch
import torchreid


class FeatureExtractor:

    def __init__(
        self,
        model_name="osnet_x1_0",
        device=None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        print("=" * 60)
        print("Loading ReID model")
        print("=" * 60)

        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            device=device
        )

        print("Device :", device)
        print("Model  :", model_name)

        print("=" * 60)

    def extract(self, image):

        feature = self.extractor(image)

        feature = feature.cpu().numpy()

        feature = feature.squeeze()

        feature = feature / np.linalg.norm(feature)

        return feature