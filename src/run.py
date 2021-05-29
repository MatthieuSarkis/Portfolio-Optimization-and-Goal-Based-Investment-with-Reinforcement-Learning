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
import time

from src.agents import Agent
from src.utilities import plot_learning_curve

class Run():
    
    def __init__(self, 
                 env: gym.Env,
                 agent: Agent,
                 n_episodes: int,
                 test_mode: bool,
                 auto_temperature: bool,
                 sac_temperature: float,
                 figure_file: str,
                 ) -> None:
        
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.test_mode = test_mode
        self.auto_temperature = auto_temperature
        self.sac_temperature = sac_temperature
        self.figure_file = figure_file
        
        if self.test_mode:
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
        
    def run(self) -> None:
        
        for i in range(self.n_episodes):
            self._run_one_episode()
                  
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
            
            if not self.test_mode:
                self.agent.learn()
                
            observation = observation_
            
        self.reward_history.append(reward)
        average_reward = np.mean(self.reward_history[-100:])
        
        self.episode += 1
        final_time = time.time()
                
        if self.auto_temperature:
            print('    episode: {:2d} | reward: {:.1f} | running_average_100: {:.1f} | step: {:6d} | {} | duration: {:.3f} | action: {}'.format(self.episode, reward, average_reward, self.step, self.agent.agent_name, final_time-initial_time, action))
    
        else: 
            print('    episode: {:2d} | reward: {:.1f} | running_average_100: {:.1f} | step: {:6d} | {} | temperature: {} | duration: {:.3f}'.format(self.episode, reward, average_reward, self.step, self.agent.agent_name, self.sac_temperature, final_time-initial_time))
        
        if average_reward > self.best_reward:
            self.best_reward = average_reward
            if not self.test_mode:
                self.agent.save_networks()
            
    def plot(self) -> None:
        
        if not self.test_mode:
            x = [i+1 for i in range(self.n_episodes)]
            plot_learning_curve(x, self.reward_history, self.figure_file)