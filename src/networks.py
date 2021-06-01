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
import os
import torch
from typing import Tuple, List

from src.utilities import make_dir

class Critic(torch.nn.Module):
    
    def __init__(self, 
                 lr_Q: float, 
                 input_shape: Tuple, 
                 layer1_neurons: int, 
                 layer2_neurons: int, 
                 action_space_dimension: Tuple, 
                 name: str, 
                 checkpoint_directory: str = 'saved_networks',
                 device: str = 'cpu',
                 ) -> None:
        
        super(Critic, self).__init__()
        self.input_shape = input_shape
        self.layer1_neurons = layer1_neurons
        self.layer2_neurons = layer2_neurons
        self.action_space_dimension = action_space_dimension
        self.name = name
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        make_dir(directory_name=checkpoint_directory)
        
        self.layer1 = torch.nn.Linear(self.input_shape[0] + action_space_dimension, self.layer1_neurons)
        self.layer2 = torch.nn.Linear(self.layer1_neurons, self.layer2_neurons)
        self.Q = torch.nn.Linear(self.layer2_neurons, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)
        
        self.device = device
        
        if torch.cuda.device_count() > 1:
            self = torch.nn.DataParallel(self)  
        self.to(device)
        
    def forward(self, 
                state: List[float], 
                action: np.array,
                ) -> torch.tensor:
        
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
                 lr_pi: float, 
                 input_shape: Tuple, 
                 layer1_neurons: int, 
                 layer2_neurons: int, 
                 max_actions: np.array, 
                 action_space_dimension: Tuple, 
                 name: str, 
                 checkpoint_directory: str = 'saved_networks',
                 device: str = 'cpu',
                 ) -> None:
        
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.layer1_neurons = layer1_neurons
        self.layer2_neurons = layer2_neurons
        self.action_space_dimension = action_space_dimension
        self.name = name
        self.max_actions = max_actions
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        make_dir(directory_name=checkpoint_directory)
        self.reparam_noise = 1e-6
        
        self.layer1 = torch.nn.Linear(*self.input_shape, self.layer1_neurons)
        self.layer2 = torch.nn.Linear(self.layer1_neurons, self.layer2_neurons)
        self.mu = torch.nn.Linear(self.layer2_neurons, self.action_space_dimension)
        self.sigma = torch.nn.Linear(self.layer2_neurons, self.action_space_dimension)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_pi)
        
        self.device = device
        
        if torch.cuda.device_count() > 1:
            self = torch.nn.DataParallel(self) 
            print("Using the GPUs!", torch.cuda.device_count())
        self.to(device)
        
    def forward(self, 
                state: List[float],
                ) -> List[torch.tensor]:
        
        x = self.layer1(state)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        
        return mu, sigma
    
    def sample_normal(self, 
                      state: List[float], 
                      reparameterize: bool = True,
                      ) -> Tuple[torch.tensor]:
        
        mu, sigma = self.forward(state)
        probabilities = torch.distributions.Normal(mu, sigma)
        
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
            
        action = torch.tanh(actions) * torch.tensor(self.max_actions).to(self.device)
        log_probabilities = probabilities.log_prob(actions)
        log_probabilities -= torch.log(1-action.pow(2) + self.reparam_noise)
        log_probabilities = log_probabilities.sum(1, keepdim=True)
        
        return action, log_probabilities
    
    def save_network_weights(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class Value(torch.nn.Module):
    
    def __init__(self, 
                 lr_Q: float, 
                 input_shape: Tuple, 
                 layer1_neurons: int, 
                 layer2_neurons: int, 
                 name: str, 
                 checkpoint_directory: str = 'saved_networks',
                 device: str = 'cpu',
                 ) -> None:
        
        super(Value, self).__init__()
        self.input_shape = input_shape
        self.layer1_neurons = layer1_neurons
        self.layer2_neurons = layer2_neurons
        self.name = name
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        make_dir(directory_name=checkpoint_directory)
        
        self.layer1 = torch.nn.Linear(*self.input_shape, self.layer1_neurons)
        self.layer2 = torch.nn.Linear(self.layer1_neurons, self.layer2_neurons)
        self.V = torch.nn.Linear(self.layer2_neurons, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)
        
        self.device = device
        
        if torch.cuda.device_count() > 1:
            self = torch.nn.DataParallel(self)             
        self.to(device)
        
    def forward(self, 
                state: List[float],
                ) -> torch.tensor:
        
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
         
class Distributional_Critic(torch.nn.Module):
    
    def __init__(self, 
                 lr_Q: float, 
                 input_shape: Tuple, 
                 layer_neurons: int, 
                 action_space_dimension: Tuple, 
                 name: str, 
                 log_std_min=-0.1, 
                 log_std_max=4,
                 checkpoint_directory: str = 'saved_networks',
                 device: str = 'cpu',
                 ) -> None:
        
        super(Distributional_Critic, self).__init__()
        self.input_shape = input_shape
        self.layer_neurons = layer_neurons
        self.name = name
        
        self.linear1 = torch.nnLinear(self.input_shape[0] + action_space_dimension, self.layer_neurons)
        self.linear2 = torch.nnLinear(self.layer_neurons, self.layer_neurons)
        self.linear3 = torch.nnLinear(self.layer_neurons, self.layer_neurons)
        self.linear_mean_4 = torch.nnLinear(self.layer_neurons, self.layer_neurons)
        self.linear_mean_5 = torch.nnLinear(self.layer_neurons, self.layer_neurons)
        self.linear_std_4 = torch.nnLinear(self.layer_neurons, self.layer_neurons)
        self.linear_std_5 = torch.nnLinear(self.layer_neurons, self.layer_neurons)

        self.mean_layer = torch.nnLinear(self.layer_neurons, 1)
        self.log_std_layer = torch.nnLinear(self.layer_neurons, 1)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.denominator = max(abs(self.log_std_min), self.log_std_max)
        
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        make_dir(directory_name=checkpoint_directory)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)
        
        self.device = device
        
        if torch.cuda.device_count() > 1:
            self = torch.nn.DataParallel(self)  
        self.to(device)

    def forward(self, 
                state: List[float], 
                action: np.array,
                ) -> Tuple[torch.Tensor]:
        
        x = torch.cat([state, action], 1)
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear3(x)
        
        x_mean = torch.nn.functional.gelu(x)
        x_mean = self.linear_mean_4(x_mean)
        x_mean = torch.nn.functional.gelu(x_mean)
        x_mean = self.linear_mean_5(x_mean)
        x_mean = torch.nn.functional.gelu(x_mean)

        x_std = torch.nn.functional.gelu(x)
        x_std = self.linear_std_4(x_std)
        x_std = torch.nn.functional.gelu(x_std)
        x_std = self.linear_std_5(x_std)
        x_std = torch.nn.functional.gelu(x_std)
        
        mean = self.mean_layer(x_mean)
        log_std = self.log_std_layer(x_std)

        log_std = torch.clamp_min(self.log_std_max*torch.tanh(log_std/self.denominator),0) + \
                  torch.clamp_max(-self.log_std_min * torch.tanh(log_std / self.denominator), 0)

        return mean, log_std

    def sample_normal(self, 
                      state: List[float], 
                      action: np.array,
                      ) -> torch.Tensor:
        
        mean, log_std = self.forward(state, action)
        std = log_std.exp()
        normal = torch.distributions.Normal(torch.zeros(mean.shape), torch.ones(std.shape))

        z = normal.sample()
        z = torch.clamp(z, -2, 2)
       
        q = mean + torch.mul(z, std)
        return q
    
    def save_network_weights(self) -> None:
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self) -> None:
        self.load_state_dict(torch.load(self.checkpoint_file))



