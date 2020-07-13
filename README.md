# Detección de anomalías en vídeos utilizando deep learning y aprendizaje no supervisado

## Introducción
En este trabajo proponemos un enfoque de **aprendizaje no supervisado** para detectar anomalías como pueden ser actos de violencia. Usamos el modelo de detección de violencia propuesto por [1] como extractor de características. El modelo original fue implementado en Pytorch [2] pero usamos una implementación con Keras y Tensorflow desarrollada por [3].


## Archivos
### Scripts de Python implementados:
- run_experiment_1.py: Ejecuta el experimento 1 y guarda los resultados.
- run_experiment_2.py: Ejecuta el experimento 2 y guarda los resultados.
- utils_svm.py: Contiene todas las funciones relacionadas con la OC-SVM.
- utils_dataset.py: Contiene funciones y clases relacionadas con los conjuntos de datos en nuestro proyecto.
- constant.py: Contiene las constantes del proyecto.

### Scripts de python importados del trabajo original
- build_model_basic.py: Construye el modelo original.
- dataset_builder.py: Contiene funciones utilizadas en "utils_dataset.py" para construir los conjuntos de datos.
- train_original_models.py: Entrena y guarda los modelos originales. Es usado para obtener los modelos pre-entrenados.


### Imágenes
- original_arquitecture.jpeg: Imagen de la arquitectura original [3].
- TFG_arquitecture: Imagen de la arquitectura propuesta en este proyecto.

### Resultados
- results_experiment_1: Directorio que contiene los resultados del experimento 1.
- results_experiment_2: Directorio que contiene los resultados del experimento 2.

## Arquitectura propuesta
![alt text](https://github.com/jrodriguez98/ViolenceDetection_CNNLSTM/blob/master/images/TFG_architecture.png)

![Architecture of the proposed model](https://github.com/jrodriguez98/ViolenceDetection_CNNLSTM/blob/master/images/TFG_architecture.png?raw=True)

## Conjuntos de datos de vídeos:
La ruta de los conjuntos se define de la siguiente forma:
- hocky - data/raw_videos/HockeyFights - [Data_Link](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89/tech)
- violentflow - data/raw_videos/violentflow - [Data_Link](https://www.openu.ac.il/home/hassner/data/violentflows/)
- movies - data/raw_videos/movies - [Data_Link](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)

Deben ser situados en **./data/raw_videos**, de esta manera:

```bash
.
├── data
    └── raw_videos
        ├── HockeyFights
        ├── movies
        └── violentflow      

```

## Modelos pre-entrenados:
Los modelos pre-entrenados se pueden descargar del siguiente Google Drive:
- https://drive.google.com/file/d/1gFx3ivUHOE03SOSr2_TF51NEkIK_dSxH/view?usp=sharing

Deben ser situados en **./models**, de esta manera:

```bash
.
├── models
    ├── hocky.h5
    ├── movies.h5
    └── violentflow.h5
    

```

## Experimentos:
### Requisitos:
- python 3.6

Ejecutar **pip install -r requirements.txt**

### Ejecutar el experimento 1
Ejecutar **python run_experiment_1.py**. Un nuevo directorio llamado *results_experiment_1* será creado junto con dos subdirectorios llamados *results_experiment_1/results_original* y *results_experiment_1/results_svm*.

### Ejecutar experimento 2
Ejecutar **python run_experiment_2.py**. Un nuevo directorio llamado *results_experiment_2* será creado junto con dos subdirectorios llamados *results_experiment_2/results_original* y *results_experiment_2/results_svm*.

## Referencias
1. Sudhakaran, Swathikiran, and Oswald Lanz. "Learning to detect violent videos
using convolution long short-term memory." In Advanced Video and Signal Based
Surveillance (AVSS), 2017 14th IEEE International Conference on, pp. 1-6. IEEE, 2017.
2. https://github.com/swathikirans/violence-recognition-pytorch
3. https://github.com/liorsidi/ViolenceDetection_CNNLSTM

------------------------

Este trabajo es un *fork* y modificación de [liorsidi/ViolenceDetection_CNNLSTM](https://github.com/liorsidi/ViolenceDetection_CNNLSTM), con el objetivo de usar los modelos como extractores de características para la Máquina de soporte vectorial de una clase (OC-SVM)

Abajo se encuentra el README original.

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
