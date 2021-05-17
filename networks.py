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

import os
from utilities import make_dir
import torch
import numpy as np

class Critic(torch.nn.Module):
    
    def __init__(self, 
                 eta1: float, 
                 input_shape: tuple, 
                 layer1_neurons: int, 
                 layer2_neurons: int, 
                 action_space_dimension: tuple, 
                 name: str, 
                 checkpoint_directory: str = 'saved_files/networks') -> None:
        
        super(Critic, self).__init__()
        self.input_shape = input_shape
        self.layer1_neurons = layer1_neurons
        self.layer2_neurons = layer2_neurons
        self.action_space_dimension = action_space_dimension
        self.name = name
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'SAC')
        make_dir(checkpoint_directory)
        
        self.layer1 = torch.nn.Linear(self.input_shape[0] + action_space_dimension, self.layer1_neurons)
        self.layer2 = torch.nn.Linear(self.layer1_neurons, self.layer2_neurons)
        self.Q = torch.nn.Linear(self.layer2_neurons, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=eta1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, 
                state: list[float], 
                action: np.array) -> torch.tensor:
        
        x = self.layer1(torch.cat([state, action], dim=1))
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        action_value = self.Q(x)
        
        return action_value

    def save_network_weights(self) -> None:
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self) -> None:
        self.load_state_dict(torch.load(self.checkpoint_file))
        
      
class Actor(torch.nn.Module):
    
    def __init__(self, 
                 eta2: float, 
                 input_shape: tuple, 
                 layer1_neurons: int, 
                 layer2_neurons: int, 
                 max_actions: np.array, 
                 action_space_dimension: tuple, 
                 name: str, 
                 checkpoint_directory: str = 'saved_files/networks') -> None:
        
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.layer1_neurons = layer1_neurons
        self.layer2_neurons = layer2_neurons
        self.action_space_dimension = action_space_dimension
        self.name = name
        self.max_actions = max_actions
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        make_dir(checkpoint_directory)
        self.reparam_noise = 1e-6 # for the clamping of the standard deviation in the forward() function
        
        self.layer1 = torch.nn.Linear(*self.input_shape, self.layer1_neurons)
        self.layer2 = torch.nn.Linear(self.layer1_neurons, self.layer2_neurons)
        self.mu = torch.nn.Linear(self.layer2_neurons, self.action_space_dimension)
        self.sigma = torch.nn.Linear(self.layer2_neurons, self.action_space_dimension)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=eta2)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, 
                state: list[float]) -> list[torch.tensor, torch.tensor]:
        
        x = self.layer1(state)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        
        return mu, sigma
    
    def sample_normal(self, 
                      state: list[float], 
                      reparameterize: bool = True) -> tuple[torch.tensor, torch.tensor]:
        
        mu, sigma = self.forward(state)
        probabilities = torch.distributions.Normal(mu, sigma)
        
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
            
        action = torch.tanh(actions) * torch.tensor(self.max_actions).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        return action, log_probs
    
    def save_network_weights(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        
        
class Value(torch.nn.Module):
    
    def __init__(self, 
                 eta1: float, 
                 input_shape: tuple, 
                 layer1_neurons: int, 
                 layer2_neurons: int, 
                 name: str, 
                 checkpoint_directory: str = 'saved_files/networks') -> None:
        
        super(Value, self).__init__()
        self.input_shape = input_shape
        self.layer1_neurons = layer1_neurons
        self.layer2_neurons = layer2_neurons
        self.name = name
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        make_dir(checkpoint_directory)
        
        self.layer1 = torch.nn.Linear(*self.input_shape, self.layer1_neurons)
        self.layer2 = torch.nn.Linear(self.layer1_neurons, self.layer2_neurons)
        self.V = torch.nn.Linear(self.layer2_neurons, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=eta1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, 
                state: list[float]) -> torch.tensor:
        
        x = self.layer1(state)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        
        value = self.V(x)
        return value
        
    def save_network_weights(self) -> None:
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self) -> None:
        self.load_state_dict(torch.load(self.checkpoint_file))
