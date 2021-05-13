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

import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
sns.set()

#***********************************************************************************

def undump_history(saving_directory='.', file=''):
    pickle_in = open(os.path.join(saving_directory, file), 'rb')
    history = pickle.load(pickle_in)
    return history

#***********************************************************************************

def plot_loss(history, save_pic=True, saving_directory='.'):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(saving_directory, 'loss_plot.png'))
    return