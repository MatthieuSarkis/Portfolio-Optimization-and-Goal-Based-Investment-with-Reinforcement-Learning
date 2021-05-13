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

import numpy as np
import os
import pandas as pd
import pickle
import preprocessing

#***********************************************************************************

def get_train_set(data_root='~/data_cfm_auction', 
                  training_input_path='input_training.csv',
                  training_output_path='output_training_IxKGwDV.csv',
                  specific_stock=False,
                  pid=360,
                  only_rev_vol=False,
                  ):
    """
    The pid is the identifier of a stock.
    We fill in the missing data.
    We focus on one stock if specific_stock=True.
    """
    
    DATA_PATH = data_root
    X_train_path = os.path.join(DATA_PATH, training_input_path)
    y_train_path = os.path.join(DATA_PATH, training_output_path)
    
    X = pd.read_csv(X_train_path)
    y = pd.read_csv(y_train_path)
    
    X, y = preprocessing.__handle_missing_values(X, y)
    
    if only_rev_vol:
        if specific_stock:
            X = X[X['pid'] == pid][['rel_vol{}'.format(i) for i in range(0, 61)]]
        else:
            X = X[['rel_vol{}'.format(i) for i in range(0, 61)]]
        y = np.exp(y.loc[X.index]).values
        X = X.values
        
    else:
        if specific_stock:
            X = X.loc(X['pid'] == pid)
        y = np.exp(y.loc[X.index]).values
        X = X.values
    
    return X, y

def get_test_set(data_root='~/data_cfm_auction', 
                 test_input_path='input_test.csv',
                 ):

    DATA_PATH = data_root
    X_test_path = os.path.join(DATA_PATH, test_input_path)
    X = pd.read_csv(X_test_path).values
    
    return X

def get_submission_example(data_root='~/data_cfm_auction', 
                           submission_example_path='submission_csv_file_random_example.csv',
                           ):

    DATA_PATH = data_root
    submission_example_path = os.path.join(DATA_PATH, submission_example_path)
    submission_example = pd.read_csv(submission_example_path)

    return submission_example

#***********************************************************************************

def dump_data(X, y, saving_directory='.'):
    pickle_out = open(os.path.join(saving_directory, 'data.pickle'), 'wb')
    pickle.dump((X, y), pickle_out)
    pickle_out.close
    return

def undump_data(data_path='./data.pickle'):
    pickle_in = open(data_path, 'rb')
    X, y = pickle.load(pickle_in)
    return X, y
    
#***********************************************************************************

if __name__ == '__main__':
    X, y = get_train_set(specific_stock=False)
    dump_data(X, y)