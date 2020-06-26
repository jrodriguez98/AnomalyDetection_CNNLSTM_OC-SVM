DATASETS_PATHS = dict(
        hocky=dict(frames='data/raw_frames/hocky', model="models/hocky.h5"),
        violentflow=dict(frames='data/raw_frames/violentflow', model="models/violentflow.h5"),
        movies=dict(frames='data/raw_frames/movies', model="models/movies.h5")
    )

FIX_LEN = 20
FIGURE_SIZE = 244
FORCE = True
BATCH_SIZE = 2