import os
from sklearn.model_selection import train_test_split
import glob
from DatasetBuilder import frame_loader, pad_sequences, get_sequences
from random import shuffle


def get_positive_class_path(datasets_frame_path: dict):
    """Return a train/test split of the paths

    Parameters:
    datasets_frame_path (dict): key: name of the dataset, value: root directory of frames

    Returns:
    train_path, test_path


    """
    video_frames_path_no = []
    video_frames_path_fight = []
    
    for dataset_name, dataset_frame_path in datasets_frame_path.items():
        for entry in os.scandir(dataset_frame_path):
            if entry.is_dir():
                complete_path = os.path.join(dataset_frame_path, entry.name)

                if dataset_name == "hocky":
                    if entry.name.startswith('no'):
                        video_frames_path_no.append(complete_path)
                    elif entry.name.startswith('fi'):
                        video_frames_path_fight.append(complete_path)

                elif dataset_name == "violentflow":
                    if entry.name.startswith('no'):
                        video_frames_path_no.append(complete_path)
                    elif entry.name.startswith('fi'):
                        video_frames_path_fight.append(complete_path)
                
                elif dataset_name == "movies":
                    if entry.name.startswith('no'):
                        video_frames_path_no.append(complete_path)
                    elif entry.name.startswith('fi'):
                        video_frames_path_fight.append(complete_path)
                
    
    train_path, test_path =  train_test_split(video_frames_path_no, test_size=0.20, random_state=42)
    
    shuffle(video_frames_path_fight) # Disorganize the figths list
    test_path.extend(video_frames_path_fight[:len(test_path)]) # 1/2 no fights and 1/2 fights

    len_test = len(test_path)
    labels_test = [0 if i < len_test/2 else 1 for i in range(len_test)]

    return train_path, test_path, labels_test


def get_sequences_x(data_paths,figure_shape,seq_length,classes=1, use_augmentation = False,use_crop=True,crop_x_y=None):
    X = []
    seq_len = 0
    for data_path in data_paths:
        frames = sorted(glob.glob(os.path.join(data_path, '*jpg')))
        x = frame_loader(frames, figure_shape)
        if(crop_x_y):
            x = [crop_img__remove_Dark(x_,crop_x_y[0],crop_x_y[1],x_.shape[0],x_.shape[1],figure_shape) for x_ in x]
        if use_augmentation:
            rand = scipy.random.random()
            corner=""
            if rand > 0.5:
                if(use_crop):
                    corner=random.choice(corner_keys)
                    x = [crop_img(x_,figure_shape,0.7,corner) for x_ in x]
                x = [frame.transpose(1, 0, 2) for frame in x]
                if(Debug_Print_AUG):
                    to_write = [list(a) for a in zip(frames, x)]
                    [cv2.imwrite(x_[0] + "_" + corner, x_[1] * 255) for x_ in to_write]

        x = [x[i] - x[i+1] for i in range(len(x)-1)]
        X.append(x)
        
    X = pad_sequences(X, maxlen=seq_length, padding='pre', truncating='pre')
    if classes > 1:
        x_ = to_categorical(x_,classes)
    return np.array(X)


def data_generator_svm(data_paths,batch_size,figure_shape,seq_length,use_aug,use_crop,crop_x_y,classes = 1):
    while True:
        indexes = np.arange(len(data_paths))
        np.random.shuffle(indexes)
        select_indexes = indexes[:batch_size]
        data_paths_batch = [data_paths[i] for i in select_indexes]

        X = get_sequences_x(train_path, figure_size, avg_length, crop_x_y=crop_x_y, classes=classes)

        yield X

def get_generators_svm(dataset_name, datasets_frames_path, fix_len, figure_size, force, classes=1, use_aug=False,
                   use_crop=True, crop_dark=None):

    train_path, test_path, test_y = get_positive_class_path(datasets_frames_path)

    if fix_len is not None:
        avg_length = fix_len
    crop_x_y = None
    if (crop_dark):
        crop_x_y = crop_dark[dataset_name]

    len_train, len_test = len(train_path), len(test_path)
    
    train_x = data_generator_svm(train_path, batch_size,figure_size,avg_length,use_aug,use_crop,crop_x_y,classes = 1)
    
    test_x, test_y = get_sequences(test_path, test_y, figure_size, avg_length, crop_x_y=crop_x_y,
                                                  classes=classes)

    print(test_x.shape)                                            

    return train_x, test_x, test_y, avg_length, len_train, len_test


# Prueba
fix_len = 20
figure_size = 244
force = True
batch_size = 2

diccionario = dict(hocky='data/raw_frames/hocky')


train_x, test_x, test_y, avg_length, len_train, len_test = get_generators_svm('hocky', diccionario, fix_len, figure_size, force, classes=1, use_aug=False,
                   use_crop=True, crop_dark=None)