
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
from utils_svm import get_model, get_positive_class_path
from utils_dataset import DatasetSequence, OCNNDatasetSequence

from build_OC_NN import OC_NN

from typing import Dict, Tuple, List, Any

from config import NUM_HIDDEN, RVALUE



def get_generators_model(dataset_name: str, dataset_frames_path: str) -> Tuple[
    DatasetSequence, DatasetSequence, List[int]]:
    """Return the generators for the pre-trained model to compute inner representations.

    Arguments:
        dataset_name: Name of the dataset used
        dataset_frame_path: Path to the frames of the dataset

    Returns:
        train_x: Generator with the training inputs for the OC-SVM.
        test_x: Generator with the test inputs inputs for the OC-SVM.
        test_y: Labels of test inputs.

    """

    from config import FIX_LEN, BATCH_SIZE, FIGURE_SIZE

    train_path, test_path, test_y = get_positive_class_path(dataset_name, dataset_frames_path)

    train_y = [1 for i in range(len(train_path))]
    train_generator = OCNNDatasetSequence(train_path, train_y, BATCH_SIZE, FIGURE_SIZE, FIX_LEN)  # Get train generator

    test_x_generator = DatasetSequence(test_path, BATCH_SIZE, FIGURE_SIZE, FIX_LEN)  # Get test generator
    test_y = np.asarray(test_y)

    len_test = len(test_path)

    assert len_test == len(test_y)

    return train_generator, test_x_generator, test_y


def train_eval_OCNN(dataset_name: str, model_path: str, frames_path: str):
    """Compute the inner representations of a dataset with a pre-trained model

    Arguments:
        dataset_name: Name of the dataset used
        model_path: Path to the model
        frames_path: Path to the frames of the dataset
        num_output_features: Number of output features of the pre-trained model

    Returns:
        train_x: Training inner representations.
        test_x: Test inner representations.
        test_y: Labels of test inputs.

    """
    optimizers = [(RMSprop, {}), (Adam, {})]

    # Take the generators from "dataset_name" path
    train_x, test_x, test_y = get_generators_model(dataset_name, frames_path)
    # Get the 'dataset_model' model
    encoder = get_model(model_path, num_output_features="flatten")

    # Create OC_NN network
    ocnn_model = OC_NN(NUM_HIDDEN, RVALUE)

    # Build the network
    ocnn_model.build(encoder=encoder)

    print("============== TRAINING THE LABELS ==========================")
    _, history = ocnn_model.fit(train_x=train_x, epochs=2, lr=1e-3)

    print("==============PREDICTING THE LABELS ==========================")
    class_report = ocnn_model.predict(test_x, test_y, history)

    return class_report


if __name__ == "__main__":

    from config import DATASETS_PATHS, FIX_LEN
    from utils_dataset import create_datasets

    create_datasets(DATASETS_PATHS, FIX_LEN)  # Extract frames from the raw videos

    for dataset_name in DATASETS_PATHS.keys():
        class_report = train_eval_OCNN(dataset_name, DATASETS_PATHS[dataset_name]['model'],
                                            DATASETS_PATHS[dataset_name]['frames'])
        print("================ CLASS REPORT ==============")
        print(class_report)
        print("============================================")
        exit()