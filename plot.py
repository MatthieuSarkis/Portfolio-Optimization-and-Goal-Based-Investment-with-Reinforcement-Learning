import numpy as np
import matplotlib.pyplot as plt
from util import maybe_make_dir

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
    