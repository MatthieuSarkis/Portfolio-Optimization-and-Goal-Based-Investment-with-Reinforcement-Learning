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

class Network(torch.nn.Module):
    """Abtract class to be inherited by the various critic and actor classes."""
    
    def __init__(self,
                 input_shape: Tuple, 
                 layer_neurons: int, 
                 network_name: str, 
                 checkpoint_directory_networks: str,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method for the Network class.
        
        Args:
            input_shape (Tuple): dimension of the state space
            layer_neurons (int): number of neurons of the various layers in the net
            name (str): name of the net
            checkpoint_directory_networks (str = 'saved_networks'): base directory for the checkpoints
            
        Returns:
            no value
        """
        
        super(Network, self).__init__()
        self.network_name = network_name
        self.checkpoint_directory_networks = checkpoint_directory_networks
        self.checkpoint_file_network = os.path.join(self.checkpoint_directory_networks, self.network_name)
         
        self.device = device
        
        self.input_shape = input_shape
        self.layer_neurons = layer_neurons
        
        if torch.cuda.device_count() > 1:
            self = torch.nn.DataParallel(self)  
        self.to(self.device)

    def forward(self, 
                *args,
                **kwargs,
                ) -> torch.tensor:

        raise NotImplementedError

    def save_network_weights(self) -> None:
        """Save checkpoint, used in training mode."""
        
        torch.save(self.state_dict(), self.checkpoint_file_network)
        
    def load_network_weights(self) -> None:
        """Load checkpoint, used in testing mode."""
        self.load_state_dict(torch.load(self.checkpoint_file_network, map_location=self.device))
        
class Critic(Network):
    """Define a critic network, whose role is to attribute a value to a (state, action) pair."""
    
    def __init__(self, 
                 lr_Q: float, 
                 action_space_dimension: Tuple, 
                 *args, 
                 **kwargs,
                 ) -> None:
        """Constructor method fo the Critic class.
        
        Args:
            lr_Q (float): learning rate for the gradient descent 
            action_space_dimension (Tuple): dimension of the action space
            
        Returns:
            no value
        """
        
        super(Critic, self).__init__(*args, **kwargs)
        self.action_space_dimension = action_space_dimension
        
        self.layer1 = torch.nn.Linear(self.input_shape[0] + action_space_dimension, self.layer_neurons)
        self.layer2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.Q = torch.nn.Linear(self.layer_neurons, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)
            
    def forward(self, 
                state: np.ndarray, 
                action: np.ndarray,
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
       
class Actor(Network):
    """Define a stochastic (Gaussian) actor, taking actions in a continous action space."""
    
    def __init__(self, 
                 lr_pi: float, 
                 max_actions: np.ndarray, 
                 action_space_dimension: Tuple, 
                 log_sigma_min: float = -20.0, 
                 log_sigma_max: float = 2.0,
                 *args,
                 **kwargs,
                 ) -> None:
        """Constructor method for the Actor class.
        
        Args:
            lr_pi (float): learning rate for the gradient descent 
            max_actions (np.array): upper (and lower) bound of the continous action space
            action_space_dimension (Tuple): dimension of the action space
            
        Returns:
            no value
        """
        
        super(Actor, self).__init__(*args, **kwargs)
        self.action_space_dimension = action_space_dimension
        self.max_actions = max_actions
        self.noise = 1e-6
        
        self.layer1 = torch.nn.Linear(*self.input_shape, self.layer_neurons)
        self.layer2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.mu = torch.nn.Linear(self.layer_neurons, self.action_space_dimension)
        self.log_sigma = torch.nn.Linear(self.layer_neurons, self.action_space_dimension)
        
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_pi)
        
    def forward(self, 
                state: np.ndarray,
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

        log_sigma = self.log_sigma(x)
        log_sigma = torch.clamp(log_sigma, min=self.log_sigma_min, max=self.log_sigma_max)       
        sigma = log_sigma.exp()
        
        return mu, sigma
    
    def sample(self, 
               state: np.ndarray, 
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
    
        
class Value(Network):
    """Define a value network, whose role is to attribute a value to a state.
    
    Used only in the first version of the Soft Actor Critic algorithm, hence in 
    the Agent_ManualTemperature class.
    """
    
    def __init__(self, 
                 lr_Q: float, 
                 *args,
                 **kwargs,
                 ) -> None:
        """Constructor method fo the Value class.
        
        Args:
            lr_Q (float): learning rate for the gradient descent 
            
        Returns:
            no value
        """
        
        super(Value, self).__init__(*args, **kwargs)        
        self.layer1 = torch.nn.Linear(*self.input_shape, self.layer_neurons)
        self.layer2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons)
        self.V = torch.nn.Linear(self.layer_neurons, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)
        
    def forward(self, 
                state: np.ndarray,
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
         
class Distributional_Critic(Network):
    """Distributional version of a critic net.
    
    Elevate the critic to a full random variable, not only considering its expectation,
    as explained in https://arxiv.org/pdf/2001.02811
    """
    
    def __init__(self, 
                 lr_Q: float, 
                 action_space_dimension: Tuple, 
                 log_sigma_min: float = -0.1, 
                 log_sigma_max: float = 5.0,
                 *args,
                 **kwargs,
                 ) -> None:
        """Constructor method fo the Distributional_Critic class.
        
        Args:
            lr_Q (float): learning rate for the gradient descent 
            action_space_dimension (Tuple): dimension of the action space
            log_sigma_min (float): clipping parameter for the log standard deviation 
            log_sigma_max (float): clipping parameter for the log standard deviation
            
        Returns:
            no value
        """
        
        super(Distributional_Critic, self).__init__(*args, **kwargs)
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
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)

    def forward(self, 
                state: List[float], 
                action: np.ndarray,
                ) -> Tuple[torch.Tensor]:
        """Implement the feedforward of the net.
        
        Args:
            state (np.array): input state 
            action (np.array): input action
            
        Returns:
            expectation and standard deviation of a 
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

        log_sigma = torch.clamp(log_sigma, min=self.log_sigma_min, max=self.log_sigma_max)
        sigma = log_sigma.exp()

        return mu, sigma

    def sample(self, 
               state: List[float], 
               action: np.ndarray,
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
        
        mu, sigma = self.forward(state, action)
                
        normal = torch.distributions.Normal(mu, sigma)
        
        if reparameterize:
            q = normal.rsample()
        else:
            q = normal.sample()
        
        return q, mu, sigma
