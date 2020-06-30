import os
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import glob
from DatasetBuilder import frame_loader, pad_sequences, get_sequences, createDataset, get_sequences_x
from random import shuffle
from keras.utils import Sequence
from keras.models import load_model, Model
import numpy as np
import pandas as pd
import math



def create_dirs():
    if not os.path.exists('data/raw_frames'):
        os.makedirs('data/raw_frames')

    if not os.path.exists('models'):
        os.makedirs('models')
    
    experiments = ['experiment_1', 'experiment_2']

    for experiment in experiments:
        if not os.path.exists(experiment):
            os.makedirs(experiment)

        path = os.path.join(experiment, 'results_svm')
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(experiment, 'results_original')
        if not os.path.exists(path):
            os.makedirs(path)


def get_positive_class_path(dataset_name, dataset_frame_path):
    """Return a train/test split of the paths

    Parameters:
    datasets_frame_path (dict): key: name of the dataset, value: root directory of frames

    Returns:
    train_path, test_path


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





class Dataset_Sequence(Sequence):

    def __init__(self, x_set, batch_size, figure_shape, seq_length, crop_x_y, classes):
        self.x = x_set
        self.batch_size = batch_size
        self.figure_shape = figure_shape
        self.seq_length = seq_length
        self.crop_x_y = crop_x_y
        self.classes = classes

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        X = get_sequences_x(batch_x, self.figure_shape, self.seq_length, crop_x_y=self.crop_x_y, classes=self.classes)

        return X

def get_generators_svm(dataset_name, dataset_frames_path, classes=1, use_aug=False,
                   use_crop=True, crop_dark=None):

    from constant import FIX_LEN, BATCH_SIZE, FIGURE_SIZE

    train_path, test_path, test_y = get_positive_class_path(dataset_name, dataset_frames_path)

    if FIX_LEN is not None:
        avg_length = FIX_LEN
    crop_x_y = None
    if (crop_dark):
        crop_x_y = crop_dark[dataset_name]

    len_train, len_test = len(train_path), len(test_path)
                            
    train_x = Dataset_Sequence(train_path, BATCH_SIZE, FIGURE_SIZE, avg_length, crop_x_y, classes=1)

    test_x = Dataset_Sequence(test_path, BATCH_SIZE, FIGURE_SIZE, avg_length, crop_x_y, classes=1)

    test_y = np.asarray(test_y) 

    assert len_test == len(test_y)                                       

    return train_x, test_x, test_y, avg_length, len_train, len_test


def get_model(dataset_model_path, num_output_features=10):

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

        intermediate_model = Model(inputs=input_layer, output=output_layer) # New model for deep representation

        return intermediate_model
    
    else:
        return model


def compute_representation_model_dataset(dataset_name, model_path, frames_path, num_output_features, get_train=True):
    # Take the generators from "dataset_name" path
    train_x, test_x, test_y, avg_length, len_train, len_test = get_generators_svm(dataset_name, frames_path, classes=1, use_aug=False,
                   use_crop=True, crop_dark=None)
    # Get the 'dataset_model' model
    model = get_model(model_path, num_output_features=num_output_features)

    if (get_train):
        train_x = model.predict_generator(train_x)
    else:
        train_x = None

    test_x = model.predict_generator(test_x)
    
    return train_x, test_x, test_y


def join_datasets (join_args):
    
    join_train_x = np.concatenate((join_args[0], join_args[3], join_args[6]), axis=0)

    join_test_x = np.concatenate((join_args[1], join_args[4], join_args[7]), axis=0)

    join_test_y = np.concatenate((join_args[2], join_args[5], join_args[8]), axis=0)

    return join_train_x, join_test_x, join_test_y


def train_eval_svm(train_x, test_x, test_y):
    
    clf = OneClassSVM(kernel='rbf', gamma='scale')
    clf.fit(train_x)

    y_pred = clf.predict(test_x) # Predictions
    
    result = classification_report(test_y, y_pred, output_dict=True)

    return result


def validate_experiment(experiment, dataset_model):

    if (experiment == 1):
        dir_experiment = 'experiment_1'
    elif(experiment == 2):
        dir_experiment = 'experiment_2'
        if(dataset_model == None):
            raise Exception('"dataset_model" can not be None in experiment 2')
    else:
        raise Exception('"num_experiment" can not be {}, possible values [1, 2]'.format(experiment))

    return dir_experiment


def eval_original_model(pred_y, test_y):

    test_y = [0 if test_y[i] == 1 else 1 for i in range(len(test_y))]
    
    result = classification_report(test_y, pred_y.round(), output_dict=True)
    
    return result


def compute_svm_experiment(experiment, num_output_features, dataset_model=None):

    from constant import DATASETS_PATHS

    dir_experiment = validate_experiment(experiment, dataset_model)
    
    join_args = []
    for dataset_name in DATASETS_PATHS.keys():
        if (experiment == 1):
            dataset_model = dataset_name

        train_x, test_x, test_y = compute_representation_model_dataset(dataset_name, DATASETS_PATHS[dataset_model]['model'], DATASETS_PATHS[dataset_name]['frames'], num_output_features)
        join_args.extend([train_x, test_x, test_y])

        if (experiment == 2 and (dataset_model == dataset_name)): # Experiment 2 just compute cross model-dataset
            continue

        result = train_eval_svm(train_x, test_x, test_y)
        pd.DataFrame(data=result, dtype=np.float).round(3).to_csv(dir_experiment + "/results_svm/dataset_{}_model_{}-{}.csv".format(dataset_name, dataset_model, num_output_features))

    
    join_train_x, join_test_x, join_test_y = join_datasets(join_args)

    del join_args[:]

    result = train_eval_svm(join_train_x, join_test_x, join_test_y)

    if (experiment == 1):
        name_join = dir_experiment + "/results_svm/dataset_join_model-{}.csv".format(num_output_features)
    else:
        name_join = dir_experiment + "/results_svm/dataset_join_model_{}-{}.csv".format(dataset_model, num_output_features)
    
    pd.DataFrame(data=result, dtype=np.float).round(3).to_csv(name_join)



def compute_original_experiment(experiment, dataset_model=None):

    from constant import DATASETS_PATHS

    dir_experiment = validate_experiment(experiment, dataset_model)

    for dataset_name in DATASETS_PATHS.keys(): 
        if (experiment == 1):
            dataset_model = dataset_name

        if (experiment == 2 and (dataset_model == dataset_name)):
            continue

        _, pred_y, test_y = compute_representation_model_dataset(dataset_name, DATASETS_PATHS[dataset_model]['model'], DATASETS_PATHS[dataset_name]['frames'], 'all', False)

        result = eval_original_model(pred_y, test_y)

        pd.DataFrame(data=result, dtype=np.float).round(3).to_csv(dir_experiment + "/results_original/dataset_{}_model_{}.csv".format(dataset_name, dataset_model))