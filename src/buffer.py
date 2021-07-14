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
from typing import Tuple, List

class ReplayBuffer():
    """Plays the role of memory for the Agents, by storing (state, action, reward, state_, done) tuples"""
    
    def __init__(self, 
                 size: int, 
                 input_shape: Tuple, 
                 action_space_dimension: int,
                 ) -> None:
        """Constructor method for the ReplayBuffer class.
        
        Args:
            size (int): maximal size of the replay buffer
            input_shape (Tuple): dimension of the observation space
            action_space_dimension (int): dimension of the action space
        
        Returns:
            no value
        """
        
        self.size = size
        self.pointer = 0
        
        self.state_buffer = np.zeros((self.size, *input_shape))
        self.new_state_buffer = np.zeros((self.size, *input_shape))
        self.action_buffer = np.zeros((self.size, action_space_dimension))
        self.reward_buffer = np.zeros(self.size)
        self.done_buffer = np.zeros(self.size, dtype=np.bool)
        
    def push(self, 
             state: np.ndarray, 
             action: np.ndarray, 
             reward: float, 
             new_state: np.ndarray, 
             done: bool,
             ) -> None:
        """Add a memory to the buffer
        
        Args:
            state (np.array): observation of the environment state 
            action (np.array): action chosen in that state
            reward (float): reward obtained for taking that action
            new_state (np.array): state in which the environment lands
            done (bool): whether one has reached the horizon or not
            
        Returns:
            no value
        """
        
        index = self.pointer % self.size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.new_state_buffer[index] = new_state
        self.done_buffer[index] = done
        
        self.pointer += 1
        
    def sample(self, 
               batch_size: int = 32,
               ) -> Tuple[np.ndarray, np.array, np.array, np.array, np.array]:
        """Sample a batch of data from the buffer.
        
        Args:
            batch_size (int): size of the batch of data to be sampled
            
        Returns:
            a tuple of np.array of memories, one memory being of the form (state, action, reward, state_, done)
        """
        
        size = min(self.pointer, self.size)
        batch = np.random.choice(size, batch_size)
        
        states = self.state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        new_states = self.new_state_buffer[batch]
        dones = self.done_buffer[batch]
        
        return states, actions, rewards, new_states, dones