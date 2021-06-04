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

import gym
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
sns.set_theme()
from sklearn.preprocessing import StandardScaler


def make_dir(directory_name: str = '') -> None: 
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
            
def plot_reward(x: np.array, 
                rewards: np.array, 
                figure_file: str, 
                mode: str,
                bins: int = 20,
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
        plt.hist(rewards, bins=bins)
        plt.title('Reward distribution')
    
    plt.savefig(figure_file) 

def instanciate_scaler(env: gym.Env,
                       mode: str) -> StandardScaler:
    
    scaler = StandardScaler()
    
    if mode == 'train':
        observation = env.reset()
        observations = [observation]
        done = False
        while not done:
            action = env.action_space.sample()
            observation_, _, done, _ = env.step(action)
            observations.append(observation_)

        scaler.fit(observations)
        with open('saved_networks/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    if mode == 'test':
        with open('saved_networks/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    return scaler

def append_corr_matrix(df: pd.DataFrame,
                       window: int,
                       ) -> pd.DataFrame:
    """
        Computes the sliding correlation matrix of a multidimensional time series, \ 
        timewise flattens it and extracts just the upper triangular part (since it is symmetric), \
        then appends it to the initial time series.
    """

    columns = ['{}/{}'.format(m, n) for (m, n) in itertools.combinations_with_replacement(df.columns, r=2)]
    corr = df.rolling(window).cov()
    corr_flattened = pd.DataFrame(index=columns).transpose()

    for i in range(df.shape[0]):

        ind = np.triu_indices(df.shape[1])
        data = corr[df.shape[1]*i : df.shape[1]*(i+1)].to_numpy()[ind]
        index = [corr.index[df.shape[1]*i][0]]

        temp = pd.DataFrame(data=data, columns=index, index=columns).transpose()
        corr_flattened = pd.concat([corr_flattened, temp])

    return pd.concat([df, corr_flattened], axis=1).iloc[window-1 : ]