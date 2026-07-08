from build_gallery import GalleryBuilder

builder = GalleryBuilder(

    video_path="videos/Video.mp4",

    csv_path="results/tracking/tracks_reid.csv",

    sample_every=5

)

gallery = builder.build()

print()

for track_id in list(gallery.keys())[:5]:

    print(

        "ID",

        track_id,

        "=>",

        len(gallery[track_id]),

        "features"

    )
