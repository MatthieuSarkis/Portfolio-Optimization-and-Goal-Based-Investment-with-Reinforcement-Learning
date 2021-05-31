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
import numpy as np
import os
import time

from src.agents import Agent
from src.utilities import plot_learning_curve, make_dir

class Run():
    
    def __init__(self, 
                 env: gym.Env,
                 agent: Agent,
                 n_episodes: int,
                 auto_temperature: bool,
                 sac_temperature: float,
                 mode: str = 'test'
                 ) -> None:
        
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.auto_temperature = auto_temperature
        self.sac_temperature = sac_temperature
        self.mode = mode
        
        if self.mode == 'test':
            self.agent.load_networks()
        
        self.step = None
        self.episode = None
        self.best_reward = None
        self.reward_history = None
        
        self._reset()
        
    def _reset(self) -> None:
        
        self.step = 0
        self.episode = 0
        self.best_reward = float('-Inf')
        self.reward_history = []
        
    def run(self,
            log_directory: str,
            ) -> None:
        
        print('>>>>> Running <<<<<')
        
        for _ in range(self.n_episodes):
            self._run_one_episode()
           
        make_dir('logs')      
        if self.auto_temperature:
            history_file = '{}/auto_temperature_{}.npy'.format(log_directory, self.mode)
        else:
            history_file = '{}/manual_temperature_{}_{}.npy'.format(log_directory, self.sac_temperature, self.mode)
            
        np.save(history_file, np.array(self.reward_history))
                 
    def _run_one_episode(self) -> None:
        
        initial_time = time.time()
        reward = 0
        done = False
        observation = self.env.reset()
        
        while not done:
            
            action = self.agent.choose_action(observation)
            observation_, reward, done, _ = self.env.step(action)
            
            if not self.auto_temperature:
                reward *= self.sac_temperature
                
            self.step += 1
            reward += reward
            
            self.agent.remember(observation, action, reward, observation_, done)
            
            if self.mode == 'train':
                self.agent.learn()
                
            observation = observation_
            
        self.reward_history.append(reward)
        average_reward = np.mean(self.reward_history[-50:])
        
        self.episode += 1
        final_time = time.time()
                
        if self.auto_temperature:
            print('    episode: {:<13d} | reward: {:<13.1f} | running_average: {:<13.1f} | {} | duration: {:<13.2f}'.format(self.episode, reward, average_reward, self.agent.agent_name, final_time-initial_time))
    
        else: 
            print('    episode: {:<13d} | reward: {:<13.1f} | running_average: {:<13.1f} | {} | temperature: {:<13.1f} | duration: {:<13.2f}'.format(self.episode, reward, average_reward, self.agent.agent_name, self.sac_temperature, final_time-initial_time))
        
        if average_reward > self.best_reward:
            self.best_reward = average_reward
            if self.mode == 'train':
                self.agent.save_networks()
            
    def plot(self,
             figure_file: str) -> None:
        
        make_dir('plots')
        figure_file = os.path.join('plots', figure_file)
        x = [i+1 for i in range(self.n_episodes)]
        plot_learning_curve(x, self.reward_history, figure_file, self.mode,
                np.sqrt(self.n_episodes).astype(int))
