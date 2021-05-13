# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#***********************************************************************************

import tensorflow as tf

#***********************************************************************************

def __create_model_RNN(input_shape=(127, 1), n_neurons=10):
    i = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(n_neurons)(i)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(i, x)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    return model

#***********************************************************************************

def __create_model_dense(input_shape=(127, 1), n_layers=2, dropout_rate=0.2):
    n_neurons = [32 * 2**i for i in range(2, n_layers+2)]
    k = 0
    i = tf.keras.layers.Input(shape=input_shape)
    x = i
    for j in range(n_layers):
        x = tf.keras.layers.Dense(n_neurons[k], activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        k += 1 
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(i, x)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    return model

#***********************************************************************************

def __rounded_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

def __create_model_RecAutoEnc():

    i = tf.keras.layers.Input(shape=[61, 2])
    x = tf.keras.layers.LSTM(100, return_sequences=True)(i)
    x = tf.keras.layers.LSTM(30)(x)

    encoded = x
    recurrent_encoder = tf.keras.models.Model(i, encoded)

    x = tf.keras.layers.RepeatVector(61)(x)
    x = tf.keras.layers.LSTM(100, return_sequences=True)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation="sigmoid"))(x)

    decoded = x
    recurrent_decoder = tf.keras.models.Model(i, decoded)

    recurrent_ae = tf.keras.models.Model(i, x)

    recurrent_ae.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.SGD(0.1), metrics=[__rounded_accuracy])

    return recurrent_ae, recurrent_encoder, recurrent_decoder

#***********************************************************************************

if __name__ == '__main__':
    recurrent_ae, recurrent_encoder, recurrent_decoder = __create_model_RecAutoEnc()
    print(recurrent_encoder.summary())