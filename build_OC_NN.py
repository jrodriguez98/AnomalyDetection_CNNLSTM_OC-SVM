from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras import backend as K
from keras.callbacks import Callback
import numpy as np

import config


class Adjust_svdd_Radius(Callback):
    def __init__(self, model, cvar, radius, X_train):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.radius = radius
        self.model = model
        self.inputs = X_train
        self.cvar = cvar

    def on_epoch_end(self, batch, logs={}):
        
        reps = self.model.predict_generator(self.inputs)
        # reps = np.reshape(reps, (len(reps), 32))
        # print("[INFO:] The OCNN - reps shape is ", reps.shape)
        self.reps = reps

        dist = np.sum((reps - self.cvar) ** 2, axis=1)
        scores = dist
        val = np.sort(scores)
        R_new = np.percentile(val, config.NU * 100)  # qth quantile of the radius.

        # R_new = np.percentile(reps, Cfg.nu * 100)  # qth quantile of the radius.
        self.rvalue = R_new
        self.radius = R_new
        # print("[INFO:] \n Updated R Value for OCNN...", self.rvalue)
        # print("[INFO:] \n Center Value used  for OCNN...", self.cvar)


class OC_NN:

    def __init__(self):
        self.Rvar = 0.0  # Radius
        self.cvar = 0.0  # center which represents the mean of the representations

    def custom_ocnn_hyperplane_loss(self):

        r = config.RVALUE
        center = self.cvar
        # w = self.oc_nn_model.layers[-2].get_weights()[0]
        # V = self.oc_nn_model.layers[-1].get_weights()[0]
        # print("Shape of w",w.shape)
        # print("Shape of V",V.shape)
        nu = config.NU

        def custom_hinge(y_true, y_pred):
            # term1 = 0.5 * tf.reduce_sum(w ** 2)
            # term2 = 0.5 * tf.reduce_sum(V ** 2)

            term3 =   K.square(r) + K.sum( K.maximum(0.0,    K.square(y_pred -center) - K.square(r)  ) , axis=1 )
            # term3 = K.square(r) + K.sum(K.maximum(0.0, K.square(r) - K.square(y_pred - center)), axis=1)
            term3 = 1 / nu * K.mean(term3)

            loss = term3

            return (loss)

        return custom_hinge

    def build(self, learning_rate, optimizer, original_model):

        # Extract the encoder
        enc_input = original_model.input
        enc_output = original_model.layers[-12].output # Flatten layer
        
        # Set up the Feed Forward NN
        x = Dense(1000, activation='relu')(enc_output)
        x = Dense(256, activation='relu')(x)
        x = Dense(10, activation='relu')(x)
        
        predictions = Dense(1, activation='sigmoid')(x)
    
        self.ocnn_model = Model(input=enc_input, outputs=predictions)  # Create the model

        opt = optimizer[0](lr=learning_rate, **optimizer[1]) 

        self.ocnn_model.compile(optimizer=optimizer, loss=self.custom_ocnn_hyperplane_loss(), metrics=['acc'])

        # Print model summary
        print(self.ocnn_model.summary())


    def fit (self, train_x):

        self.reps = self.ocnn_model.predict_generator(train_x)
        
        # consider the value all the number of batches (and thereby samples) to initialize from
        c = np.mean(self.reps, axis=0)

        eps= 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        # If c_i is too close to 0 in dimension i, set to +-eps.
        self.cvar = c  # Initialize the center


        out_batch = Adjust_svdd_Radius(self.ocnn_model, self.cvar, self.Rvar, train_x)
        callbacks = [out_batch]

        y_reps = out_batch.reps

        # fit the  model_svdd by defining a custom loss function
        H = self.ocnn_model.fit_generator(train_x, y_reps, shuffle=True,
                                epochs=100,
                                verbose=0,
                                callbacks=callbacks
                                )

        