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



import numpy as np
import matplotlib.pyplot as plt
from utilities import maybe_make_dir

def plot(mode):
    images_folder = 'rl_trader_images'
    maybe_make_dir(images_folder)
    
    a = np.load('rl_trader_rewards/{}.npy'.format(mode))
    
    if mode == 'train':
        plt.plot(a)
    else:
        plt.hist(a, bins=20)
        
    plt.title(mode)
    plt.savefig('{}/{}.png'.format(images_folder, mode))
    