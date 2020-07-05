import os
import cv2
import pickle
import math
from keras.utils import Sequence
from DatasetBuilder import get_sequences_x, save_figures_from_video
from typing import Dict



class Dataset_Sequence(Sequence):
    """

    """

    def __init__(self, x_set, batch_size, figure_shape, seq_length):
        self.x = x_set
        self.batch_size = batch_size
        self.figure_shape = figure_shape
        self.seq_length = seq_length

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size) # Generator lenght

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = get_sequences_x(batch_x, self.figure_shape, self.seq_length)

        return X


def create_dataset(dataset_name: str, dataset_video_path: str, dataset_frames_path: str, fix_len: int):
    """Extract the frames from the raw videos of a dataset.

    Arguments:
        dataset_name: Name of the dataset used
        dataset_video_path: Path to the raw videos
        dataset_frames_path: Path where frames will be saved
        fix_len: Number of input frames of the pre-trained model
        get_train: Indicate if it is needed the training samples.

    """

    #Extract images for each video for each dataset
    if not os.path.exists(dataset_frames_path):
        os.makedirs(dataset_frames_path)
    
    for filename in os.listdir(dataset_video_path):
        if filename.endswith(".avi") or filename.endswith(".mpg"):
            video_images_file = os.path.join(dataset_frames_path,filename[:-4], 'video_summary.pkl')
            video_images = save_figures_from_video(dataset_video_path, filename[:-4],filename[-4:], dataset_frames_path, fix_len =fix_len)
            if dataset_name == "hocky":
                if filename.startswith("fi"):
                    video_images['label'] = 1
            elif dataset_name == "violentflow":
                if "violence" in filename:
                    video_images['label'] = 1
            elif dataset_name == "movies":
                if "fi" in filename:
                    video_images['label'] = 1
            with open(video_images_file, 'wb') as f:
                pickle.dump(video_images, f, pickle.HIGHEST_PROTOCOL)

def create_datasets(datasets_paths: Dict[str, str], fix_len: int) -> None:
    """Extract the frames from the raw videos of all datasets.

    Arguments:
        datasets_paths: Dictionary with videos path and frames path for each dataset
        fix_len: Number of input frames of the pre-trained model.

    """

    if not os.path.exists('data/raw_frames'):
        os.makedirs('data/raw_frames')

        for dataset_name, dataset_paths in datasets_paths.items():
            create_dataset(dataset_name, dataset_paths['videos'], dataset_paths['frames'], fix_len)

    

if __name__ == "__main__":
    from constant import DATASETS_PATHS, FIX_LEN

    create_datasets(DATASETS_PATHS, FIX_LEN)

