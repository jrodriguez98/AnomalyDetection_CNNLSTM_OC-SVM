import tensorflow as tf

from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Flatten

from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from utils_dataset import DatasetSequence

from sklearn.metrics import classification_report

import numpy as np

from config import FIX_LEN, BATCH_SIZE, FIGURE_SIZE, NU
from loss import quantile_loss

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sess = tf.Session()
K.set_session(sess)


class Adjust_svdd_Radius(Callback):
    def __init__(self, model, cvar, radius, train_path, rep_dim):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.radius = radius
        self.model = model
        self.inputs = DatasetSequence(train_path, BATCH_SIZE, FIGURE_SIZE, FIX_LEN)
        self.cvar = cvar
        self.reps = np.zeros((len(train_path), rep_dim))

    def on_epoch_end(self, batch, logs={}):
        self.reps = self.model.predict_generator(self.inputs)
        # reps = np.reshape(reps, (len(reps), 32))
        # print("[INFO:] The OCNN - reps shape is ", reps.shape)

        scores = np.sum((self.reps - self.cvar) ** 2, axis=1)

        val = np.sort(scores)

        r_new = np.percentile(val, NU * 100)  # qth quantile of the radius.

        # R_new = np.percentile(reps, Cfg.nu * 100)  # qth quantile of the radius.
        self.radius = r_new
        # Corrección: self.model.radius = r_new
        # print("[INFO:] \n Updated R Value for OCNN...", self.rvalue)
        # print("[INFO:] \n Center Value used  for OCNN...", self.cvar)


class OC_NN:

    def __init__(self, hidden_size, r=1.0):
        self.hidden_size = hidden_size
        self.r = r  # Radius
        self.ocnn_model = None
        self.w = None
        self.V = None

    def get_predictions(self, scores):
        assert ((len(scores) % 2) == 0)
        num_anomalies = len(scores) / 2

        sorted_keys = np.argsort(scores)

        normal_index = sorted_keys[:num_anomalies]
        anomalies_index = sorted_keys[-num_anomalies:]

        assert (len(normal_index) == anomalies_index)

        y_pred = np.ones(len(scores), dtype=int)
        y_pred[anomalies_index] = -1

        return y_pred, normal_index, anomalies_index

    def get_predictions(self, scores):

        assert ((len(scores) % 2) == 0)
        num_anomalies = len(scores) / 2

        sorted_keys = np.argsort(scores)

        normal_index = sorted_keys[:num_anomalies]
        anomalies_index = sorted_keys[-num_anomalies:]

        assert (len(normal_index) == anomalies_index)

        y_pred = np.ones(len(scores), dtype=int)
        y_pred[anomalies_index] = -1

        return y_pred, normal_index, anomalies_index

    def custom_ocnn_loss(self, nu, w, V):
        def custom_hinge(_, y_pred):
            loss = 0.5 * tf.reduce_sum(w ** 2) + 0.5 * tf.reduce_sum(V ** 2) + quantile_loss(self.r, y_pred, nu)
            self.r = tf.contrib.distributions.percentile(tf.reduce_max(y_pred, axis=1), q=100 * nu)
            return loss

        return custom_hinge

    def build(self, encoder):

        encoder_model = Model(encoder.input, encoder.layers[-1].output)

        self.ocnn_model = Sequential()

        for layer in encoder_model.layers:
            self.ocnn_model.add(layer)

        # Define the layer from input to hidden units
        input_hidden = Dense(
            self.hidden_size,
            kernel_initializer='glorot_normal',
            name='input_hidden')

        self.ocnn_model.add(input_hidden)
        self.ocnn_model.add(Activation('linear'))

        # Define the layer from hidden to output
        hidden_out = Dense(1, name="hidden_output")

        self.ocnn_model.add(hidden_out)
        self.ocnn_model.add(Activation('sigmoid'))

        """x = Dense(512, activation='linear', name='dense_FCN_1', use_bias=False)(encoder.layers[-1].output)
        predictions = Dense(32, activation='linear', name='dense_FCN_3', use_bias=False)(x)"""

        with sess.as_default():
            self.w = input_hidden.get_weights()[0]
            self.V = hidden_out.get_weights()[0]

        # Print model summary
        print(self.ocnn_model.summary())

        """opt = optimizer[0](lr=learning_rate, **optimizer[1])
        self.ocnn_model.compile(optimizer=opt, loss=self.custom_ocnn_hyperplane_loss(), metrics=['acc'])"""


    def fit(self, train_x, epochs, lr, save=False):

        def r_metric(*args):
            return self.r

        r_metric.__name__ = 'r'

        def quantile_loss_metric(*args):
            return quantile_loss(self.r, args[1], NU)

        quantile_loss_metric.__name__ = 'quantile_loss'

        self.ocnn_model.compile(
            optimizer=Adam(lr=lr, decay=lr / epochs),
            loss=self.custom_ocnn_loss(NU, self.w, self.V),
            metrics=[r_metric, quantile_loss_metric]
        )

        # despite the fact that we don't have a ground truth `y`, the fit function requires a label argument,
        # so we just supply a dummy vector of 0s
        type(train_x)
        history = self.ocnn_model.fit_generator(
            train_x,
            epochs=epochs,
            use_multiprocessing=False
        )

        if save:
            import os
            from datetime import datetime
            if not os.path.exists('onnn_models'):
                os.mkdir('onnn_models')
            model_dir = f"models/ocnn_{datetime.now().strftime('%Y-%m-%d-%H:%M:%s')}"
            os.mkdir(model_dir)
            with sess.as_default():
                w = self.ocnn_model.get_layer("input_hidden").get_weights()[0]
                V = self.ocnn_model.get_layer("hidden_output").get_weights()[0]
            self.ocnn_model.save(f"{model_dir}/model.h5")
            np.savez(f"{model_dir}/params.npz", w=w, V=V, nu=NU)

        return self.ocnn_model, history

    def load_model(self, model_dir):
        """
        loads a pretrained model
        :param model_dir: directory where model and model params (w, V, and nu) are saved
        :param nu: same as nu described in train_model
        :return: loaded model
        """
        params = np.load(f'{model_dir}/params.npz')
        w = params['w']
        V = params['V']
        nu = params['nu'].tolist()
        self.onnn_model = load_model(f'{model_dir}/model.h5',
                           custom_objects={'custom_hinge': self.custom_ocnn_loss(nu, w, V)})
        return self.onnn_model

    def predict(self, test_x, test_y, history):

        y_pred = self.ocnn_model.predict_generator(test_x)

        r = history.history['r'].pop()

        # Get the anomalies based on the decision score of the paper
        anomalies_index = [i for i in range(len(y_pred)) if (y_pred[i, 0] - r < 0)]

        y_pred = np.ones(len(y_pred), dtype=int)
        y_pred[anomalies_index] = -1

        class_report = classification_report(test_y, y_pred, output_dict=True)

        return class_report
