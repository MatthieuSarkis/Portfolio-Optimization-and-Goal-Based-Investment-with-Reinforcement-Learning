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
    """Define a critic network, whose role is to attribute a value to a (state, action) pair."""
    
    def __init__(self, 
                 lr_Q: float, 
                 input_shape: Tuple, 
                 layer_neurons: int, 
                 action_space_dimension: Tuple, 
                 name: str, 
                 checkpoint_directory: str = 'saved_networks',
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method fo the Critic class.
        
        Args:
            lr_Q (float): learning rate for the gradient descent 
            input_shape (Tuple): dimension of the state space
            layer_neurons (int): number of neurons of the various layers in the net
            action_space_dimension (Tuple): dimension of the action space
            name (str): name of the net
            checkpoint_directory (str = 'saved_networks'): base directory for the checkpoints
            device (str = 'cpu'): cpu or gpu
            
        Returns:
            no value
        """
        
        super(Critic, self).__init__()
        self.input_shape = input_shape
        self.layer_neurons = layer_neurons
        self.action_space_dimension = action_space_dimension
        self.name = name
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        make_dir(directory_name=checkpoint_directory)
        
        self.layer1 = torch.nn.Linear(self.input_shape[0] + action_space_dimension, self.layer_neurons)
        self.layer2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.Q = torch.nn.Linear(self.layer_neurons, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)
        
        self.device = device
        
        if torch.cuda.device_count() > 1:
            self = torch.nn.DataParallel(self)  
        self.to(device)
        
    def forward(self, 
                state: np.array, 
                action: np.array,
                ) -> torch.tensor:
        """Implement the feedforward of the net.
        
        Args:
            state (np.array): input state 
            action (np.array): input action
            
        Returns:
            value attributed to the (state, action) pair
        """
        
        x = self.layer1(torch.cat([state, action], dim=1))
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        action_value = self.Q(x)
        
        return action_value

    def save_network_weights(self) -> None:
        """Save checkpoint, used in training mode."""
        
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self) -> None:
        """Load checkpoint, used in testing mode."""
        
        self.load_state_dict(torch.load(self.checkpoint_file))
        
      
class Actor(torch.nn.Module):
    """Define a stochastic (Gaussian) actor, taking actions in a continous action space."""
    
    def __init__(self, 
                 lr_pi: float, 
                 input_shape: Tuple, 
                 layer_neurons: int, 
                 max_actions: np.array, 
                 action_space_dimension: Tuple, 
                 name: str, 
                 log_sigma_min: float = -0.1, 
                 log_sigma_max: float = 4.0,
                 checkpoint_directory: str = 'saved_networks',
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method for the Actor class.
        
        Args:
            lr_pi (float): learning rate for the gradient descent 
            input_shape (Tuple): dimension of the state space
            layer_neurons (int): number of neurons of the various layers in the net
            max_actions (np.array): upper (and lower) bound of the continous action space
            action_space_dimension (Tuple): dimension of the action space
            name (str): name of the net
            checkpoint_directory (str): base directory path for checkpoints
            device (str): cpu or gpu
            
        Returns:
            no value
        """
        
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.layer_neurons = layer_neurons
        self.action_space_dimension = action_space_dimension
        self.name = name
        self.max_actions = max_actions
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        make_dir(directory_name=checkpoint_directory)
        self.noise = 1e-6
        
        self.layer1 = torch.nn.Linear(*self.input_shape, self.layer_neurons)
        self.layer2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.mu = torch.nn.Linear(self.layer_neurons, self.action_space_dimension)
        self.log_sigma = torch.nn.Linear(self.layer_neurons, self.action_space_dimension)
        
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.denominator = max(abs(self.log_sigma_min), self.log_sigma_max)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_pi)
        self.device = device
        
        if torch.cuda.device_count() > 1:
            self = torch.nn.DataParallel(self) 
            print("Using the GPUs!", torch.cuda.device_count())
        self.to(device)
        
    def forward(self, 
                state: np.array,
                ) -> List[torch.tensor]:
        """Implement the feedforward of the net.
        
        Args:
            state (np.array): input state in which the actor has to pick an action
            
        Returns:
            expectation and standard deviation of a Normal distribution
        """
        
        x = self.layer1(state)
        x = torch.nn.functional.gelu(x)
        x = self.layer2(x)
        x = torch.nn.functional.gelu(x)
           
        mu = self.mu(x)

        #sigma = self.log_sigma(x)
        #sigma = torch.clamp(sigma, min=self.noise, max=1)
        
        log_sigma = self.log_sigma(x)
        log_sigma = torch.clamp(log_sigma, min=-20, max=2)       
        sigma = log_sigma.exp()
        
        return mu, sigma
    
    def sample(self, 
               state: np.array, 
               reparameterize: bool = True,
               ) -> Tuple[torch.tensor]:
        """Sample from the Normal distribution, output of feedforward method, to give an action
        
        Args:
            state (np.array): state of the environment in which the actor has to pick an action
            reparameterize (bool): whether one should sample using the reparemeterization trick or not
        
        Returns:
            action sampled from Normal disribution, as well as log probability of the given action
        """
        
        mu, sigma = self.forward(state)
        
        normal = torch.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = normal.rsample()
        else:
            actions = normal.sample()
            
        action = torch.tanh(actions) * torch.tensor(self.max_actions).to(self.device)
        log_probabilities = normal.log_prob(actions)
        log_probabilities -= torch.log(1-action.pow(2) + self.noise)
        log_probabilities = log_probabilities.sum(1, keepdim=True)
        
        return action, log_probabilities
    
    def save_network_weights(self):
        """Save checkpoint, used in training mode."""
        
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self):
        """Load checkpoint, used in testing mode."""
        
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class Value(torch.nn.Module):
    """Define a value network, whose role is to attribute a value to a state.
    
    Used only in the first version of the Soft Actor Critic algorithm, hence in 
    the Agent_ManualTemperature class.
    """
    
    def __init__(self, 
                 lr_Q: float, 
                 input_shape: Tuple, 
                 layer_neurons: int, 
                 name: str, 
                 checkpoint_directory: str = 'saved_networks',
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method fo the Value class.
        
        Args:
            lr_Q (float): learning rate for the gradient descent 
            input_shape (Tuple): dimension of the state space
            layer_neurons (int): number of neurons of the various layers in the net
            name (str): name of the net
            checkpoint_directory (str = 'saved_networks'): base directory for the checkpoints
            device (str = 'cpu'): cpu or gpu
            
        Returns:
            no value
        """
        
        super(Value, self).__init__()
        self.input_shape = input_shape
        self.layer_neurons = layer_neurons
        self.name = name
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        make_dir(directory_name=checkpoint_directory)
        
        self.layer1 = torch.nn.Linear(*self.input_shape, self.layer_neurons)
        self.layer2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.V = torch.nn.Linear(self.layer_neurons, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)
        
        self.device = device
        
        if torch.cuda.device_count() > 1:
            self = torch.nn.DataParallel(self)             
        self.to(device)
        
    def forward(self, 
                state: np.array,
                ) -> torch.tensor:
        """Implement the feedforward of the net.
        
        Args:
            state (np.array): input state to which one wants to attribute a value
            
        Returns:
            value attributed to the input state
        """
        
        x = self.layer1(state)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        
        value = self.V(x)
        return value
        
    def save_network_weights(self) -> None:
        """Save checkpoint, used in training mode."""
        
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self) -> None:
        """Load checkpoint, used in testing mode."""
        
        self.load_state_dict(torch.load(self.checkpoint_file))
         
class Distributional_Critic(torch.nn.Module):
    """Distributional version of a critic net.
    
    Elevate the critic to a full random variable, not only considering its expectation,
    as explained in https://arxiv.org/pdf/2001.02811
    """
    
    def __init__(self, 
                 lr_Q: float, 
                 input_shape: Tuple, 
                 layer_neurons: int, 
                 action_space_dimension: Tuple, 
                 name: str, 
                 log_sigma_min: float = -0.1, 
                 log_sigma_max: float = 4.0,
                 checkpoint_directory: str = 'saved_networks',
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method fo the Distributional_Critic class.
        
        Args:
            lr_Q (float): learning rate for the gradient descent 
            input_shape (Tuple): dimension of the state space
            layer_neurons (int): number of neurons of the various layers in the net
            action_space_dimension (Tuple): dimension of the action space
            name (str): name of the net
            log_sigma_min (float): clipping parameter for the log standard deviation 
            log_sigma_max (float): clipping parameter for the log standard deviation
            checkpoint_directory (str = 'saved_networks'): base directory for the checkpoints
            device (str = 'cpu'): cpu or gpu
            
        Returns:
            no value
        """
        
        super(Distributional_Critic, self).__init__()
        self.input_shape = input_shape
        self.layer_neurons = layer_neurons
        self.name = name
        
        self.linear1 = torch.nn.Linear(self.input_shape[0] + action_space_dimension, self.layer_neurons)
        self.linear2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.linear3 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.linear_mu_1 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.linear_mu_2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.linear_mu_3 = torch.nn.Linear(self.layer_neurons, 1)
        self.linear_log_sigma_1 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.linear_log_sigma_2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.linear_log_sigma_3 = torch.nn.Linear(self.layer_neurons, 1)
        
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.denominator = max(abs(self.log_sigma_min), self.log_sigma_max)
        
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
        """Implement the feedforward of the net.
        
        Args:
            state (np.array): input state 
            action (np.array): input action
            
        Returns:
            expectation and log standard deviation of a 
            normal critic-value distribution attributed to the (state, action) input pair
        """
        
        x = self.linear1(torch.cat([state, action], dim=1))
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear3(x)
        x = torch.nn.functional.gelu(x)
        
        mu = self.linear_mu_1(x)
        mu = torch.nn.functional.gelu(mu)
        mu = self.linear_mu_2(mu)
        mu = torch.nn.functional.gelu(mu)
        mu = self.linear_mu_3(mu)

        log_sigma = self.linear_log_sigma_1(x)
        log_sigma = torch.nn.functional.gelu(log_sigma)
        log_sigma = self.linear_log_sigma_2(log_sigma)
        log_sigma = torch.nn.functional.gelu(log_sigma)
        log_sigma = self.linear_log_sigma_3(log_sigma)

        log_sigma = torch.clamp_min(self.log_sigma_max*torch.tanh(log_sigma/self.denominator),0) + \
                    torch.clamp_max(-self.log_sigma_min * torch.tanh(log_sigma / self.denominator), 0)

        return mu, log_sigma

    def sample(self, 
               state: List[float], 
               action: np.array,
               reparameterize: bool = True,
               ) -> torch.Tensor:
        """Sample from the Normal distribution, output of feedforward method, to give a critic-value
        
        Args:
            state (np.array): state of the environment
            action (np.array): action taken in the state
            reparameterize (bool): whether one should sample using the reparemeterization trick or not
        
        Returns:
            critic-value sampled from Normal disribution
            expectation of the critic-value random variable 
            standard deviation of the critic-value random variable
        """
        
        mu, log_sigma = self.forward(state, action)
        sigma = log_sigma.exp()
                
        normal = torch.distributions.Normal(mu, sigma)
        
        if reparameterize:
            q = normal.rsample()
        else:
            q = normal.sample()
        
        return q, mu, sigma
    
    def save_network_weights(self) -> None:
        """Save checkpoint, used in training mode."""
        
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_network_weights(self) -> None:
        """Load checkpoint, used in testing mode."""
        
        self.load_state_dict(torch.load(self.checkpoint_file))
