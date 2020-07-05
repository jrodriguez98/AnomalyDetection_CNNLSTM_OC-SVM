# Anomaly detection in videos using deep learning and unsupervised learning

## Introduction
In this work we propose an **unsupervised anomaly detection** approach to detect anomalies in videos such as violence. We use the violence detection model proposed by [1] as features extractor. The original model was implemented with Pytorch [2] but we use Keras and TensorFlow implementation developed by [3].


## Architecture structure
![alt text](https://github.com/jrodriguez98/ViolenceDetection_CNNLSTM/tree/master/images/TFG_architecture.png)


## Video datasets paths:
data path are defined as follows:
- hocky - data/raw_videos/HockeyFights - [Data_Link](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89/tech)
- violentflow - data/raw_videos/violentflow - [Data_Link](https://www.openu.ac.il/home/hassner/data/violentflows/)
- movies - data/raw_videos/movies - [Data_Link](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)

## Experiments:
### Running experiment 1
Run ''python run_experiment_1.py''. A new directory called ''experiment_1'' will be created along with 2 subdirectories called ''experiment_1/results_original'' and ''experiment_1/results_svm''

### Running experiment 2
Run ''python run_experiment_2.py''. A new directory called ''experiment_2'' will be created along with 2 subdirectories called ''experiment_2/results_original'' and ''experiment_2/results_svm''

## References
1. Sudhakaran, Swathikiran, and Oswald Lanz. "Learning to detect violent videos
using convolution long short-term memory." In Advanced Video and Signal Based
Surveillance (AVSS), 2017 14th IEEE International Conference on, pp. 1-6. IEEE, 2017.
2. https://github.com/swathikirans/violence-recognition-pytorch
3. https://github.com/liorsidi/ViolenceDetection_CNNLSTM

------------------------

This is a fork and modification of [liorsidi/ViolenceDetection_CNNLSTM](https://github.com/liorsidi/ViolenceDetection_CNNLSTM), in order to use the models as features extractor for a One-Class Support Vector Machine (OC-SVM)

Below you will find the original README.

------------------------

# Learning to Detect Violent Videos using Convolution LSTM

This work is based on violence detection model proposed by [1] with minor modications.
The original model was implemented with Pytorch [2] while in this work we implement it with Keras and TensorFlow as a back-end. 
The model incorporates pre-trained convolution Neural Network (CNN) connected to Convolutional LSTM (ConvLSTM) layer.
The model takes as an inputs the raw video, converts it into frames and output a binary classication of violence or non-violence label.

### Architecture structure
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/Architecture.jpeg)


## Running configurations
### Video datasets paths:
data path are defined as follows:
- hocky - data/raw_videos/HockeyFights - [Data_Link](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89/tech)
- violentflow - data/raw_videos/violentflow - [Data_Link](https://www.openu.ac.il/home/hassner/data/violentflows/)
- movies - data/raw_videos/movies - [Data_Link](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)

### Libraries perquisites:
- python 2.7
- numpy 1.14.0
- keras 2.2.0
- tensorflow 1.9.0
- Pillow 3.1.2
- opencv-python 3.4.1.15

### Running operation:
just run python run.py
(currently we don't support arguments from command line)

## Results
#### Hyper-tuning results (Hocky data)
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/hyperparameters_results.JPG)

#### Hockey dataset results
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/Hockey_results.png)

## Refrences
1. Sudhakaran, Swathikiran, and Oswald Lanz. "Learning to detect violent videos
using convolution long short-term memory." In Advanced Video and Signal Based
Surveillance (AVSS), 2017 14th IEEE International Conference on, pp. 1-6. IEEE, 2017.
2. https://github.com/swathikirans/violence-recognition-pytorch
