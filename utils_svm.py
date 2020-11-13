import os
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from random import shuffle
import numpy as np
from utils_dataset import DatasetSequence
from typing import Dict, Tuple, List, Any
import json

from keras.models import load_model, Model


def create_dirs_experiment(experiment: int) -> None:
    """Create the experiment directories

    It creates the directories needed in order to store the results

    Parameters:
        experiment (int): Number of the experiment.

    """

    dir_experiment = validate_experiment(experiment)

    if not os.path.exists(dir_experiment):
        os.makedirs(dir_experiment)

    path = os.path.join(dir_experiment, 'results_svm')
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(dir_experiment, 'results_original')
    if not os.path.exists(path):
        os.makedirs(path)


def get_positive_class_path(dataset_name: str, dataset_frame_path: str) -> Tuple[List[str], List[str], List[int]]:
    """Return a train/test split of the paths

    Arguments:
        dataset_name : Name of the dataset used
        dataset_frame_path: Path to the frames of the dataset

    Returns:
        train_path: Paths used as training dataset (only positive class)
        test_path: Paths used as test dataset (both classes)
        labels_test: Labels of the test set.
    
    """
    video_frames_path_no = []
    video_frames_path_fight = []
    
    
    for entry in os.scandir(dataset_frame_path):
        if entry.is_dir():
            complete_path = os.path.join(dataset_frame_path, entry.name)

            if dataset_name == "hocky":
                if entry.name.startswith('fi'):
                    video_frames_path_fight.append(complete_path)
                else:
                    video_frames_path_no.append(complete_path)

            elif dataset_name == "violentflow":
                if "violence" in entry.name:
                    video_frames_path_fight.append(complete_path)
                else:
                    video_frames_path_no.append(complete_path)
            
            elif dataset_name == "movies":
                if "fi" in entry.name:
                    video_frames_path_fight.append(complete_path)
                else:
                    video_frames_path_no.append(complete_path)
                
    
    train_path, test_path =  train_test_split(video_frames_path_no, test_size=0.20, random_state=42)
    
    shuffle(video_frames_path_fight) # Disorganize the figths list
    test_path.extend(video_frames_path_fight[:len(test_path)]) # 1/2 no fights and 1/2 fights

    len_test = len(test_path)
    labels_test = [1 if i < len_test/2 else -1 for i in range(len_test)]

    return train_path, test_path, labels_test


def get_generators_model(dataset_name: str, dataset_frames_path: str) -> Tuple[DatasetSequence, DatasetSequence, List[int]]:
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
   
    train_x = DatasetSequence(train_path, BATCH_SIZE, FIGURE_SIZE, FIX_LEN) # Get train generator  

    test_x = DatasetSequence(test_path, BATCH_SIZE, FIGURE_SIZE, FIX_LEN) # Get test generator

    test_y = np.asarray(test_y)

    len_test = len(test_path)

    assert len_test == len(test_y)                                       

    return train_x, test_x, test_y


def get_model(dataset_model_path: str, num_output_features: int=10) -> Model:
    """Return the model with the a number of output features

    Arguments:
        dataset_model_path: Path to the pre-trained model
        num_output_features: string or number indicating the number of output features

    Returns:
        model: Neural network model

    """

    cut_model = True

    if (num_output_features == 10):
        index = -4
    elif (num_output_features == 256):
        index = -7
    elif (num_output_features == 1000):
        index = -9
    elif (num_output_features == "flatten"):
        index = -12
    elif (num_output_features == "all"):
        cut_model = False
    else:
        raise Exception('num_output_features can not be {}, possible values [10, 256, 1000]'.format(num_output_features))

    model = load_model(dataset_model_path)

    if (cut_model):
        input_layer = model.input
        output_layer = model.layers[index].output

        model = Model(inputs=input_layer, output=output_layer) # New model for deep representation

        return model

    else:
        return model


def compute_representation_model_dataset(dataset_name: str, model_path: str, frames_path: str, 
                                         num_output_features: Any, get_train: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the inner representations of a dataset with a pre-trained model

    Arguments:
        dataset_name: Name of the dataset used
        model_path: Path to the model
        frames_path: Path to the frames of the dataset
        num_output_features: Number of output features of the pre-trained model
        get_train: Indicate if it is needed the training samples.

    Returns:
        train_x: Training inner representations.
        test_x: Test inner representations.
        test_y: Labels of test inputs.

    """
    # Take the generators from "dataset_name" path
    train_x, test_x, test_y = get_generators_model(dataset_name, frames_path)
    # Get the 'dataset_model' model
    model = get_model(model_path, num_output_features=num_output_features)

    if (get_train):
        train_x = model.predict_generator(train_x)
    else:
        train_x = None

    test_x = model.predict_generator(test_x)
    
    return train_x, test_x, test_y




def join_datasets (join_args: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 'join' dataset out of the representations of the others

    Arguments:
        join_args: List containing the representations of the three datasets

    Returns:
        join_train_x: join training dataset.
        join_test_x: join test dataset.
        join_test_y: join test labels.

    """
    
    join_train_x = np.concatenate((join_args[0], join_args[3], join_args[6]), axis=0)

    join_test_x = np.concatenate((join_args[1], join_args[4], join_args[7]), axis=0)

    join_test_y = np.concatenate((join_args[2], join_args[5], join_args[8]), axis=0)

    return join_train_x, join_test_x, join_test_y


def train_eval_svm(train_x: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> Dict[str, Any]:
    """Train and evaluate the OC-SVM with the inner representations.

    Arguments:
        train_x: Training dataset.
        test_x: Test dataset.
        test_y: Test labels.

    Returns:
        result: Result over test dataset.

    """

    clf = OneClassSVM(kernel='rbf', gamma='scale')
    clf.fit(train_x)

    y_pred = clf.predict(test_x) # Predictions

    result = classification_report(test_y, y_pred, output_dict=True)

    return result


def validate_experiment(experiment: int) -> str:
    """Validate the number of experiment introduced
    
    Arguments:
        experiment: Number of experiment

    Returns:
        dir_experiment: Name of the experiment directory.

    """

    if (experiment == 1):
        dir_experiment = 'results_experiment_1'
    elif(experiment == 2):
        dir_experiment = 'results_experiment_2'
    else:
        raise Exception('"num_experiment" can not be {}, possible values [1, 2]'.format(experiment))

    return dir_experiment

    
def save_json (full_path: str, data: Dict[Any, Any]) -> None:
    """Save a dictionary to .json file

    Arguments:
        full_path: Path where file will be saved
        data: Data to save

    """
    
    with open(full_path, 'w') as file:
        json.dump(data, file, indent=4)

def eval_original_model(pred_y: np.ndarray, test_y: np.ndarray) -> Dict[str, Any]:
    """Evaluate original model predictions

    Arguments:
        pred_y: Predictions of the dataset.
        test_y: True labels of the dataset.

    Returns:
        result: Result over test dataset.

    """

    test_y = [0 if test_y[i] == 1 else 1 for i in range(len(test_y))]
    
    result = classification_report(test_y, pred_y.round(), output_dict=True)
    
    return result


def compute_svm_experiment(experiment: int, num_output_features: Any, dataset_model: str=None) -> None:
    """Performs the experiment tasks related with the OC-SVM

    Arguments:
        experiment: Number of experiment
        num_output_features: Number of output features desired
        dataset_model: Name of the pre-trained model

    """

    from config import DATASETS_PATHS

    dir_experiment = validate_experiment(experiment)
    
    join_args = []
    for dataset_name in DATASETS_PATHS.keys():
        if (experiment == 1):
            dataset_model = dataset_name

        train_x, test_x, test_y = compute_representation_model_dataset(dataset_name, DATASETS_PATHS[dataset_model]['model'], DATASETS_PATHS[dataset_name]['frames'], num_output_features)
        join_args.extend([train_x, test_x, test_y])

        if (experiment == 2 and (dataset_model == dataset_name)): # Experiment 2 just compute cross model-dataset
            continue

        result = train_eval_svm(train_x, test_x, test_y)
        save_json(dir_experiment + "/results_svm/dataset_{}_model_{}-{}.json".format(dataset_name, dataset_model, num_output_features), result)

    
    join_train_x, join_test_x, join_test_y = join_datasets(join_args)

    del join_args[:]

    result = train_eval_svm(join_train_x, join_test_x, join_test_y)

    if (experiment == 1):
        name_join = dir_experiment + "/results_svm/dataset_join_model-{}.json".format(num_output_features)
    else:
        name_join = dir_experiment + "/results_svm/dataset_join_model_{}-{}.json".format(dataset_model, num_output_features)
    
    save_json(name_join, result)



def compute_original_experiment(experiment: int, dataset_model: Any =None) -> None:
    """Performs the experiment tasks related with the original model

    Arguments:
        experiment: Number of experiment
        dataset_model: Name of the pre-trained model
    
    """

    from config import DATASETS_PATHS

    dir_experiment = validate_experiment(experiment)

    for dataset_name in DATASETS_PATHS.keys(): 
        if (experiment == 1):
            dataset_model = dataset_name

        if (experiment == 2 and (dataset_model == dataset_name)):
            continue

        _, pred_y, test_y = compute_representation_model_dataset(dataset_name, DATASETS_PATHS[dataset_model]['model'], DATASETS_PATHS[dataset_name]['frames'], 'all', False)

        result = eval_original_model(pred_y, test_y)

        save_json(dir_experiment + "/results_original/dataset_{}_model_{}.json".format(dataset_name, dataset_model), result)



"""def compute_OCNN_experiment(experiment: int, dataset_model: str) -> None:
    Performs the experiment tasks related with the OC-NN

    Arguments:
        experiment: Number of experiment
        dataset_model: Name of the pre-trained model
    

    from config import DATASETS_PATHS

    dir_experiment = validate_experiment(experiment)
    
    join_args = []
    for dataset_name in DATASETS_PATHS.keys():
        if (experiment == 1):
            dataset_model = dataset_name

        train_x, test_x, test_y = compute_representation_model_dataset(dataset_name, DATASETS_PATHS[dataset_model]['model'], DATASETS_PATHS[dataset_name]['frames'], num_output_features)
        join_args.extend([train_x, test_x, test_y])

        if (experiment == 2 and (dataset_model == dataset_name)): # Experiment 2 just compute cross model-dataset
            continue

        result = train_eval_svm(train_x, test_x, test_y)
        save_json(dir_experiment + "/results_svm/dataset_{}_model_{}-{}.json".format(dataset_name, dataset_model, num_output_features), result)

    
    join_train_x, join_test_x, join_test_y = join_datasets(join_args)

    del join_args[:]

    result = train_eval_svm(join_train_x, join_test_x, join_test_y)

    if (experiment == 1):
        name_join = dir_experiment + "/results_svm/dataset_join_model-{}.json".format(num_output_features)
    else:
        name_join = dir_experiment + "/results_svm/dataset_join_model_{}-{}.json".format(dataset_model, num_output_features)
    
    save_json(name_join, result)"""

