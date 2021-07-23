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
from sklearn.preprocessing import StandardScaler

from src.agents import Agent
from src.logger import Logger

class Run():
    """Main class to run the training or the testing."""
    
    def __init__(self, 
                 env: gym.Env,
                 agent: Agent,
                 n_episodes: int,
                 agent_type: str,
                 scaler: StandardScaler,
                 checkpoint_directory: str,
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
        self.checkpoint_directory = checkpoint_directory
        
        if self.mode == 'test':
            self.agent.load_networks()
        
        self.step = None
        self.episode = None
        self.best_reward = None
        
        self.logger = Logger(mode=self.mode,
                             checkpoint_directory=self.checkpoint_directory)
        
        self._reset()
        
    def _reset(self) -> None:
        """Initialize the environment and the reward history."""
        
        self.step = 0
        self.episode = 0
        self.best_reward = float('-Inf')
        
    def run(self) -> None:
        """Run the training or the testing during a certain number of steps."""
        
        print('>>>>> Running <<<<<\n')
        
        for _ in range(self.n_episodes):
            self._run_one_episode()
                 
    def _run_one_episode(self) -> None:
        """Agent takes one step in the environment, and learns if in train mode."""
        
        self.logger.set_time_stamp(1)
        
        reward: float = 0
        done: bool = False
        observation = self.env.reset()
        observation = self.scaler.transform([observation])[0]
        
        # initializing a list to keep track of the porfolio value during the episode
        if self.mode == 'test':
            portfolio_value_history = [self.env._get_portfolio_value()]
        
        while not done:
            
            action = self.agent.choose_action(observation)
            observation_, reward, done, _ = self.env.step(action)
            observation_ = self.scaler.transform([observation_])[0]
            
            # rescale the reward to account for the relative normalization between the 
            # expected return and the entropy term in the loss function
            if self.agent_type == 'manual_temperature':
                reward *= self.sac_temperature
                
            self.step += 1
            reward += reward
            
            if self.mode == 'test':
                portfolio_value_history.append(self.env._get_portfolio_value())
            
            self.agent.remember(observation, action, reward, observation_, done)
            
            if self.mode == 'train':
                self.agent.learn(self.step)
                
            observation = observation_
             
        self.logger.logs["reward_history"].append(reward)
        average_reward = np.mean(self.logger.logs["reward_history"][-50:])
        
        if self.mode == 'test':
            self.logger.logs["portfolio_value_history_of_histories"].append(portfolio_value_history)
        
        self.episode += 1
        
        self.logger.set_time_stamp(2)
        self.logger.print_status(self.episode)
        
        if average_reward > self.best_reward:
            self.best_reward = average_reward
            if self.mode == 'train':
                self.agent.save_networks()
            
        