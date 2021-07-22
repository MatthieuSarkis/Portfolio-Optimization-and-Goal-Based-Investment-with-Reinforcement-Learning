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
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

from src.agents import Agent
from src.utilities import plot_reward

class Run():
    """Main class to run the training or the testing."""
    
    def __init__(self, 
                 env: gym.Env,
                 agent: Agent,
                 n_episodes: int,
                 agent_type: str,
                 scaler: StandardScaler,
                 checkpoint_directory_logs: str,
                 sac_temperature: float = 1.0,
                 mode: str = 'test',
                 ) -> None:
        """Constructor method of the class Run.
        
        Args:
            env (gym.Env): trading environment in which the agent evolves
            agent (Agent): Soft Actor Critic like agent
            n_episodes (int): total number of episodes for training, or testing (since the policy is stochastic)
            agent_type (str): name of the type of agent for saving files
            scaler (StandardScaler): already fitted sklearn standard scaler, used as a preprocessing step
            sac_temperature (float): rescaling factor of rewards, in case one uses manual type agent
            mode (str): train or test
            
        Returns:
            no value
        """
        
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.agent_type = agent_type
        self.sac_temperature = sac_temperature
        self.mode = mode
        self.scaler = scaler
        self.checkpoint_directory_logs = checkpoint_directory_logs
        
        if self.mode == 'test':
            self.agent.load_networks()
        
        self.step = None
        self.episode = None
        self.best_reward = None
        self.reward_history = None
        self.portfolio_history_of_histories = None
        
        self._reset()
        
    def _reset(self) -> None:
        """Initialize the environment and the reward history."""
        
        self.step = 0
        self.episode = 0
        self.best_reward = float('-Inf')
        self.reward_history = []
        
    def run(self) -> None:
        """Run the training or the testing during a certain number of steps.
        
        Args:
            log_directory (str): filepath where to save the reward history as a numpy array
            
        Returns:
            no value
        """
        
        print('>>>>> Running <<<<<')
        
        for _ in range(self.n_episodes):
            self._run_one_episode()
                   
        np.save(os.path.join(self.checkpoint_directory_logs, self.mode), np.array(self.reward_history))
                 
    def _run_one_episode(self) -> None:
        """Agent takes one step in the environment, and learns if in train mode."""
        
        initial_time = time.time()
        reward: float = 0
        done: bool = False
        observation = self.env.reset()
        observation = self.scaler.transform([observation])[0]
        
        while not done:
            
            action = self.agent.choose_action(observation)
            observation_, reward, done, _ = self.env.step(action)
            observation_ = self.scaler.transform([observation_])[0]
            
            if self.agent_type == 'manual_temperature':
                reward *= self.sac_temperature
                
            self.step += 1
            reward += reward
            
            self.agent.remember(observation, action, reward, observation_, done)
            
            if self.mode == 'train':
                self.agent.learn(self.step)
                
            observation = observation_
            
        self.reward_history.append(reward)
        average_reward = np.mean(self.reward_history[-50:])
        
        self.episode += 1
        final_time = time.time()
        
        print('    episode: {:<13d} | reward: {:<13.1f} | duration: {:<13.2f}'.format(self.episode, reward, final_time-initial_time))

        if average_reward > self.best_reward:
            self.best_reward = average_reward
            if self.mode == 'train':
                self.agent.save_networks()
            
    def plot(self,
             checkpoint_directory_plots: str) -> None:
        """Call a helper function to plot the reward history in train mode and the reward distribution in test mode.
        
        Args:
            figure_file (str): hte filepath where to save the plot file
            
        Returns:
            no value
        """
        
        figure_file = os.path.join(checkpoint_directory_plots, self.mode)
        x = [i+1 for i in range(self.n_episodes)]
        plot_reward(x, self.reward_history, figure_file, self.mode, np.sqrt(self.n_episodes).astype(int))
