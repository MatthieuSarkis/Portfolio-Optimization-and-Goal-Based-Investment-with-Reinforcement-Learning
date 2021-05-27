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
        self.best_reward = None
        self.reward_history = None
        
        self._reset()
        
    def _reset(self) -> None:
        
        self.step = 0
        self.best_reward = float('-Inf')
        self.reward_history = []
        
    def run(self) -> None:
        
        for i in range(self.n_episodes):
        
            reward = 0
            done = False
            observation = self.env.reset()
            
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, _ = self.env.step(action)
                if not self.auto_temperature:
                    reward *= self.sac_temperature
                self.steps += 1
                reward += reward
                self.agent.remember(observation, action, reward, observation_, done)
                if not self.test_mode:
                    self.learn()
                observation = observation_
                
            self.reward_history.append(reward)
            avg_reward = np.mean(self.reward_history[-100:])
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                if not self.test_mode:
                    self.agent.save_networks()
                    
            if self.auto_temperature:
                print('episode:', i, 
                      'reward: %.1f' % reward, 
                      'running_average_100_episodes: %.1f' % avg_reward,
                      'step: %d' % self.steps, self.agent.agent_name)
        
            else: 
                print('episode:', i, 
                      'reward: %.1f' % reward, 
                      'running_average_100: %.1f' % avg_reward,
                      'step: %d' % self.steps, self.agent.agent_name, 
                      'temperature:', self.sac_temperature)
            
    def plot(self) -> None:
        
        if not self.test_mode:
            x = [i+1 for i in range(self.n_episodes)]
            plot_learning_curve(x, self.reward_history, self.figure_file)