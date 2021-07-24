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

"""
This class remains to be implemented to handles logging more efficiently.
"""

import numpy as np
import os
import time

from src.utilities import plot_reward, plot_portfolio_value
class Logger():
    
    def __init__(self,
                 mode: str,
                 checkpoint_directory: str,
                 ) -> None:
        
        self.mode = mode
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_directory_logs = os.path.join(self.checkpoint_directory, "logs")
        self.checkpoint_directory_plots = os.path.join(self.checkpoint_directory, "plots")
        
        self.logs: dict = {}
        self.logs['reward_history'] = []
        
        if self.mode =='test':
            self.logs['portfolio_value_history_of_histories'] = []
    
        self.time_stamp = [0, 0]
            
    def set_time_stamp(self,
                   i: int,
                   ) -> None:
        
        self.time_stamp[i-1] = time.time()
                            
    def print_status(self,
                     episode: int,
                     ) -> None:
        
        print('    episode: {:<13d} | reward: {:<13.1f} | duration: {:<13.2f}'.format(episode, self.logs["reward_history"][-1], self.time_stamp[1]-self.time_stamp[0]))
    
    def save_logs(self) -> None:
        """Saves all the necessary logs to 'checkpoint_directory_logs' directory."""
        
        reward_history_array = np.array(self.logs['reward_history'])
        np.save(os.path.join(self.checkpoint_directory_logs, self.mode+"_reward_history.npy"), reward_history_array)
        
        if self.mode =='test':
            portfolio_value_history_of_histories_array = np.array(self.logs['portfolio_value_history_of_histories'])
            np.save(os.path.join(self.checkpoint_directory_logs, self.mode+"_portfolio_value_history.npy"), portfolio_value_history_of_histories_array)
        
    def generate_plots(self) -> None:
        """Call a helper function to plot the reward history in train mode and the reward distribution in test mode."""
        
        checkpoint_directory_plots = os.path.join(self.checkpoint_directory, "plots")
        
        reward_history_array = np.array(self.logs['reward_history'])
        n_episodes = reward_history_array.shape[0]
        episodes = [i+1 for i in range(n_episodes)]
        plot_reward(x=episodes, 
                    rewards=reward_history_array, 
                    figure_file=os.path.join(checkpoint_directory_plots, self.mode+"_reward"), 
                    mode=self.mode, 
                    bins=np.sqrt(n_episodes).astype(int))
        
        if self.mode =='test':
            portfolio_value_history_of_histories_array = np.array(self.logs['portfolio_value_history_of_histories'])
            n_days = portfolio_value_history_of_histories_array.shape[1]
            days = [i+1 for i in range(n_days)]
            plot_portfolio_value(x=days, 
                                 values=portfolio_value_history_of_histories_array, 
                                 figure_file=os.path.join(checkpoint_directory_plots, self.mode+"_portfolioValue"))
      