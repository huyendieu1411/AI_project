from collections import defaultdict

import numpy as np


class FeatureGallery:

    def __init__(self):

        self.gallery = defaultdict(list)

    def add(self, track_id, feature):

        self.gallery[int(track_id)].append(feature)

    def get(self, track_id):

        return self.gallery[int(track_id)]

    def ids(self):

        return list(self.gallery.keys())