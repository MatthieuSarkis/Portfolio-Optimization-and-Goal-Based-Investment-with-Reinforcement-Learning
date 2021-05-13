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

from argparse import ArgumentParser
from datetime import datetime
import get_data
import numpy as np
import os
import tensorflow as tf
import train
import utilities

#***********************************************************************************

def main(args, print_args=True):
    
    start_time = datetime.now()

    # Initialize the seed.
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    
    # Get the data
    print('\n Loading the data... \n')
    if os.path.exists(args.data_path) and args.dumped_data:
        X, y = get_data.undump_data(args.data_path)
    else:
        X, y = get_data.get_train_set(specific_stock=args.specific_stock, 
                                      pid=args.pid, 
                                      only_rev_vol=args.only_rev_val)    

    saving_directory = utilities.make_dirs(odir=args.odir, 
                                           folder_name=utilities.time_to_string(start_time))

    if print_args:
        print(60*'*')
        dic_args = vars(args)
        for key in dic_args:
            print('# {} = {}'.format(key, dic_args[key]))
        print('# X.shape={}  y.shape={}'.format(X.shape, y.shape))
        print(60*'*')
        print('\n')

    train.train(X, y, 
                saving_directory=saving_directory, 
                random_state=args.random_state,
                patience=args.patience,
                test_size=args.test_size,
                epochs=args.epochs,
                n_gpus=args.n_gpus,
                batch_size=args.batch_size,  
                n_layers=args.n_layers,  
                dropout_rate=args.dropout_rate,  
                for_RNN=args.for_RNN, 
                preprocess=args.preprocess,                          
                )

    end_time = datetime.now()
    
    print('\n')
    print(60*'*')
    print ('\n # start_time={} end_time={} elpased_time={}'.format(utilities.time_to_string(start_time), 
                                                                utilities.time_to_string(end_time), 
                                                                end_time - start_time))

#***********************************************************************************

if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--dumped_data', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, default='./data.pickle')
    parser.add_argument('--specific_stock', action='store_true', default=False)
    parser.add_argument('--only_rev_vol', action='store_true', default=False)
    parser.add_argument('--pid', type=int, default=360)
    parser.add_argument('--odir', type=str, default='saved_files')
    parser.add_argument('--random_state', action='store', type=int, default=None)
    parser.add_argument('--test_size', type=float, default=0.20)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', action='store', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--for_RNN', action='store_true', default=False)
    parser.add_argument('--preprocess', action='store_true', default=False)
    
    args = parser.parse_args()
    main(args)