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

import os
import neural_networks
import pickle
import plot_performance
import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings

#***********************************************************************************

def __print_model_summary(model, saving_directory):
    """
    Prints the model architecture both on the standard output and in a log file.
    """
    
    print(model.summary())
    print('\n')
    
    with open(os.path.join(saving_directory, 'model_summary.log'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
#***********************************************************************************

def __create_and_compile_model(input_shape):
    """
    Actually I decided to compile the model directly in the function defining it 
    in neural_networks.py
    """

    model = neural_networks.create_model(input_shape)          
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    return model

#***********************************************************************************

def __define_callbacks(use_learning_rate_scheduler=True,
                       use_earlystopping=True,
                       use_checkpoint=True,
                       saving_directory=".",
                       patience=1):
    """
    Returns a list of callback for the fit method of the model.
    """
    
    callbacks = []
    
    if use_earlystopping:
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience, 
                                          restore_best_weights=True)
        callbacks += [early_stopping_cb]

    if use_learning_rate_scheduler:
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=10e-2, patience=5)
        callbacks += [lr_scheduler]

    if use_checkpoint:
        checkpoint_file = os.path.join(saving_directory, "checkpoint.h5")
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, 
                                        save_best_only=True, 
                                        monitor='val_loss',
                                        save_weights_only=True) 
        callbacks += [checkpoint_cb]
        
    return callbacks

#***********************************************************************************

def train(X, y,
          random_state=42,
          test_size=0.20,
          saving_directory='.',
          n_gpus=1,
          patience=1,
          epochs=10,
          batch_size=None,
          print_model_summary=True,
          use_learning_rate_scheduler=True,
          use_checkpoint=True,
          use_earlystopping=True,
          use_tensorboard=False,
          dumped_history=True,
          save_model=True,
          plot_loss=True,
          n_layers=3,
          dropout_rate=0.2,
          for_RNN=False,
          preprocess=False,
         ):

    T = X.shape[1]
    
    input_shape=(T,)
    if for_RNN:
        X = X.reshape(-1, T, 1)
        input_shape=(T, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state, 
                                                        ) 
    
    if preprocess:
        preprocessing.__standardize_data(X_train, X_test)
    
    # If some GPUs are available then use them, otherwise simply use the CPUs.
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(device_type)
    if len(devices) > 1 and n_gpus > 1 :
        #devices_names = [d.name.split('e:')[1] for d in devices]
        #strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = neural_networks.__create_model_dense(input_shape=input_shape, n_layers=n_layers, dropout_rate=dropout_rate)
            
    else:
        model = neural_networks.__create_model_dense(input_shape=input_shape, n_layers=n_layers, dropout_rate=dropout_rate)

    # define a few callbacks.
    callbacks = __define_callbacks(use_learning_rate_scheduler=use_learning_rate_scheduler,
                                   use_earlystopping=use_earlystopping,
                                   use_checkpoint=use_checkpoint,
                                   use_tensorboard=use_tensorboard,
                                   saving_directory=saving_directory,
                                   patience=patience)
    
    # print model info
    if print_model_summary:
        __print_model_summary(model, saving_directory)
        
    # training the model
    history = model.fit(X_train, y_train,  
                        validation_data=(X_test, y_test), 
                        callbacks=callbacks,
                        epochs=epochs,
                        batch_size=batch_size)

    if save_model:
        model_path = os.path.join(saving_directory, 'saved_model.h5')
        model.save(model_path)

    if dumped_history:
        pickle_out = open(os.path.join(saving_directory, 'dumped_history'), 'wb')
        pickle.dump(history.history, pickle_out)
        pickle_out.close
        
    if plot_loss:
        plot_performance.plot_loss(history, save_pic=True, saving_directory=saving_directory)

    return model, history
   
#***********************************************************************************
