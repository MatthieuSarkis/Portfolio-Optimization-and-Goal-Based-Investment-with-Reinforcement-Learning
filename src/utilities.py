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

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set_theme()


def make_dir(directory_name: str = '') -> None: 
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
            
def plot_learning_curve(x: np.array, 
                        rewards: np.array, 
                        figure_file: str, 
                        mode: str,
                        ) -> None:
    
    running_average = np.zeros(len(rewards))
    for i in range(len(running_average)):
        running_average[i] = np.mean(rewards[max(0, i-50): i+1])
        
    if mode == 'train':
        plt.plot(x, rewards, linestyle='-', color='blue', label='reward')
        plt.plot(x, running_average, linestyle='--', color='green', label='running average 50')
        plt.legend()
        plt.title('Reward as a function of the epoch/episode')
        
    elif mode == 'test':
        plt.hist(rewards, bins=20)
        plt.title('Reward distribution')
    
    plt.savefig(figure_file) 