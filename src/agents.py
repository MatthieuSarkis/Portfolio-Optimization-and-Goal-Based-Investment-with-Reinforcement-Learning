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
import torch
from typing import Tuple

from src.buffer import ReplayBuffer
from src.networks import Actor, Critic, Value, Distributional_Critic
   
class Agent():
    
    def __init__(self,
                 lr_Q: float, 
                 lr_pi: float, 
                 input_shape: Tuple, 
                 tau: float, 
                 env: gym.Env, 
                 agent_name: str, 
                 action_space_dimension: int, 
                 gamma: float = 0.99, 
                 size: int = 1000000,
                 layer_size: int = 256, 
                 batch_size: int = 256,
                 device: str = 'cpu',
                 ) -> None:
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(size, input_shape, action_space_dimension)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.action_space_dimension = action_space_dimension
        self.lr_Q = lr_Q
        self.lr_pi = lr_pi
        self.env = env
        self.agent_name = agent_name
        self.layer_size = layer_size
        self.device = device

        self.actor = Actor(self.lr_pi, 
                           self.input_shape, 
                           self.layer_size, 
                           action_space_dimension=self.action_space_dimension, 
                           name=self.agent_name+'_actor',
                           max_actions=self.env.action_space.high,
                           device=self.device)
        
        self._network_list = [self.actor]
        self._targeted_network_list = []
    
    def remember(self, 
                 state: np.array, 
                 action: np.array, 
                 reward: float, 
                 new_state: np.array, 
                 done: bool,
                 ) -> None:
        
        self.memory.push(state, action, reward, new_state, done)
            
    @staticmethod   
    def _initialize_weights(net: torch.nn.Module) -> None:
        
        if type(net) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(net.weight)
            net.bias.data.fill_(1e-2)
    
    def _update_target_networks(self, 
                                tau: float = None,
                                ) -> None:
    
        if tau is None:
            tau = self.tau

        shift = len(self._targeted_network_list) // 2

        for i in range(shift):
            
            target_params = self._targeted_network_list[i+shift].named_parameters()
            params = self._targeted_network_list[i].named_parameters()
            
            target_params = dict(target_params)
            params = dict(params)
            
            for name in params:
                params[name] = tau * params[name].clone() + (1 - tau) * target_params[name].clone()
                
            self._targeted_network_list[i+shift].load_state_dict(params)

    def choose_action(self, 
                      observation: np.array,
                      ) -> np.array:
        
        state = torch.Tensor([observation]).to(self.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        
        return actions.cpu().detach().numpy()[0]

    def save_networks(self) -> None:
        
        print(' ... saving networks ... ')    
        for network in self._network_list:
            network.save_network_weights()
        
    def load_networks(self) -> None:
        
        print(' ... loading networks ... ')
        for network in self._network_list:
            network.load_network_weights()
   
class Agent_ManualTemperature(Agent):
    
    def __init__(self, 
                 *args, 
                 **kwargs,
                 ) -> None:
        
        super(Agent_ManualTemperature, self).__init__(*args, **kwargs)
        
        self.critic_1 = Critic(self.lr_Q, 
                               self.input_shape, 
                               self.layer_size, 
                               action_space_dimension=self.action_space_dimension, 
                               name=self.agent_name+'_critic1',
                               device=self.device)
        
        self.critic_2 = Critic(self.lr_Q, 
                               self.input_shape, 
                               self.layer_size, 
                               action_space_dimension=self.action_space_dimension, 
                               name=self.agent_name+'_critic2',
                               device=self.device)
        
        self.value = Value(self.lr_Q, 
                           self.input_shape, 
                           self.layer_size,
                           name=self.agent_name+'_value',
                           device=self.device)
        
        self.target_value = Value(self.lr_Q, 
                                  self.input_shape, 
                                  self.layer_size, 
                                  name=self.agent_name+'_target_value',
                                  device=self.device)
        
        self._network_list += [self.critic_1, self.critic_2, self.value, self.target_value]
        self._targeted_network_list += [self.value, self.target_value]
        
        for network in self._network_list:
            network.apply(self._initialize_weights) 
        
        self._update_target_networks(tau=1)
 
    def learn(self) -> None:
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        # VALUE UPDATE
        value = self.value(states).view(-1)
        value_ = self.target_value(states_).view(-1)
        value_[dones] = 0.0
        
        actions, log_probabilities = self.actor.sample_normal(states, reparameterize=False)
        log_probabilities = log_probabilities.view(-1)
                
        q1_new_policy = self.critic_1.forward(states, actions)
        q2_new_policy = self.critic_2.forward(states, actions)
            
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        
        value_target = critic_value - log_probabilities
        value_loss = 0.5 * torch.nn.functional.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True) 
        
        self.value.optimizer.step()
        
        # POLICY UPDATE
        actions, log_probabilities = self.actor.sample_normal(states, reparameterize=True)
        log_probabilities = log_probabilities.view(-1)
        
        q1_new_policy = self.critic_1.forward(states, actions)
        q2_new_policy = self.critic_2.forward(states, actions)
        
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probabilities - critic_value
        actor_loss = torch.mean(actor_loss)
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        
        # CRITIC UPDATE
        q_target = rewards + self.gamma * value_
        
        q1 = self.critic_1.forward(states, actions).view(-1)
        q2 = self.critic_2.forward(states, actions).view(-1)
        
        critic_1_loss = 0.5 * torch.nn.functional.mse_loss(q1, q_target)
        critic_2_loss = 0.5 * torch.nn.functional.mse_loss(q2, q_target)
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        critic_loss = critic_1_loss + critic_2_loss
        
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        # EXPONENTIALLY SMOOTHED COPY TO THE TARGET VALUE NETWORK
        self._update_target_network()
           
class Agent_AutomaticTemperature(Agent):
    
    def __init__(self, 
                 lr_alpha: float,
                 alpha: float = 1.0,
                 *args,
                 **kwargs,
                 ) -> None:
        
        super(Agent_AutomaticTemperature, self).__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.action_space_dimension).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True).to(self.device).detach().requires_grad_(True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
        self.critic_1 = Critic(self.lr_Q, 
                               self.input_shape, 
                               self.layer_size,
                               action_space_dimension=self.action_space_dimension, 
                               name=self.agent_name+'_critic1',
                               device=self.device)
        
        self.critic_2 = Critic(self.lr_Q, 
                               self.input_shape, 
                               self.layer_size,
                               action_space_dimension=self.action_space_dimension, 
                               name=self.agent_name+'_critic2',
                               device=self.device)
        
        self.target_critic_1 = Critic(self.lr_Q, 
                                      self.input_shape, 
                                      self.layer_size, 
                                      action_space_dimension=self.action_space_dimension, 
                                      name=self.agent_name+'_target_critic1',
                                      device=self.device)
        
        self.target_critic_2 = Critic(self.lr_Q, 
                                      self.input_shape, 
                                      self.layer_size, 
                                      action_space_dimension=self.action_space_dimension, 
                                      name=self.agent_name+'_target_critic2',
                                      device=self.device)
        
        self._network_list += [self.critic_1, self.critic_2, self.target_critic_1, self.target_critic_2]
        self._targeted_network_list += [self.critic_1, self.critic_2, self.target_critic_1, self.target_critic_2]
        
        for network in self._network_list:
            network.apply(self._initialize_weights) 
        
        self._update_target_networks(tau=1)
        
    def learn(self) -> None:
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
               
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        # CRITIC UPDATE
        actions_, log_probabilities_ = self.actor.sample_normal(states_, reparameterize=False)
        
        q1_ = self.target_critic_1.forward(states_, actions_)
        q2_ = self.target_critic_2.forward(states_, actions_)
        
        target_soft_value_ = (torch.min(q1_, q2_) - (self.alpha * log_probabilities_)).view(-1)
        target_soft_value_[dones] = 0
        q_target = rewards + self.gamma * target_soft_value_
        
        q1 = self.critic_1.forward(states, actions).view(-1)
        q2 = self.critic_2.forward(states, actions).view(-1)
        
        critic_1_loss = 0.5 * torch.nn.functional.mse_loss(q1, q_target)
        critic_2_loss = 0.5 * torch.nn.functional.mse_loss(q2, q_target)
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward(retain_graph=True)
        
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        # POLICY UPDATE
        actions, log_probabilities = self.actor.sample_normal(states, reparameterize=True)
        log_probabilities = log_probabilities
        
        q1_ = self.target_critic_1.forward(states, actions)
        q2_ = self.target_critic_2.forward(states, actions)
        
        critic_value = torch.min(q1_, q2_)
        
        actor_loss = self.alpha * log_probabilities - critic_value
        actor_loss = torch.mean(actor_loss.view(-1))
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
                    
        # TEMPERATURE UPDATE
        log_alpha_loss = -(self.log_alpha * (log_probabilities + self.target_entropy).detach()).mean()
        
        self.log_alpha_optimizer.zero_grad()
        log_alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # EXPONENTIALLY SMOOTHED COPY TO THE TARGET CRITIC NETWORKS
        self._update_target_networks()
        
        
class Distributional_Agent(Agent):
    
    def __init__(self, 
                 lr_alpha: float,
                 alpha: float = 1.0,
                 *args,
                 **kwargs,
                 ) -> None:
        
        super(Agent_AutomaticTemperature, self).__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.action_space_dimension).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True).to(self.device).detach().requires_grad_(True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
        self.critic = Distributional_Critic(self.lr_Q,
                                            self.input_shape,
                                            self.layer_size,
                                            self.action_space_dimension,
                                            self.agent_name,
                                            log_std_min=-0.1,
                                            log_std_max=4,
                                            checkpoint_directory=self.agent_name+'_critic',
                                            device=self.device)
        
        self.target_critic = Distributional_Critic(self.lr_Q,
                                                   self.input_shape,
                                                   self.layer_size,
                                                   self.action_space_dimension,
                                                   self.agent_name,
                                                   log_std_min=-0.1,
                                                   log_std_max=4,
                                                   checkpoint_directory=self.agent_name+'_critic',
                                                   device=self.device)
        
        self.target_actor = Actor(self.lr_pi, 
                                  self.input_shape, 
                                  self.layer_size, 
                                  action_space_dimension=self.action_space_dimension, 
                                  name=self.agent_name+'_target_actor',
                                  max_actions=self.env.action_space.high,
                                  device=self.device)
        
        self._network_list += [self.critic, self.target_critic, self.target_actor]
        self._targeted_network_list += [self.critic, self.actor, self.target_critic, self.target_actor]
        
        for network in self._network_list:
            network.apply(self._initialize_weights) 
        
        self._update_target_networks(tau=1)
        
    def learn(self) -> None:
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
               
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        # CRITIC UPDATE
        
        # to do
        
        ###
        actions_, log_probabilities_ = self.actor.sample_normal(states_, reparameterize=False)
        
        q_ = self.target_critic.forward(states_, actions_)        
        target_soft_value_ = (q_ - self.alpha * log_probabilities_).view(-1)
        target_soft_value_[dones] = 0
        
        q_target = rewards + self.gamma * target_soft_value_
        q = self.critic.forward(states, actions).view(-1)
        critic_loss = torch.nn.functional.mse_loss(q, q_target)
        
        self.critic.optimizer.zero_grad()  
        critic_loss.backward(retain_graph=True)
        self.critic.optimizer.step()
        ###
        
        # POLICY UPDATE
        actions, log_probabilities = self.actor.sample_normal(states, reparameterize=True)
        log_probabilities = log_probabilities
        
        critic_value = self.target_critic.sample(states, actions, reparameterize=True)
        # In their article they don't use the target critic here...
        
        actor_loss = self.alpha * log_probabilities - critic_value
        actor_loss = torch.mean(actor_loss.view(-1))
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
                    
        # TEMPERATURE UPDATE
        log_alpha_loss = -(self.log_alpha * (log_probabilities + self.target_entropy).detach()).mean()
        
        self.log_alpha_optimizer.zero_grad()
        log_alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # EXPONENTIALLY SMOOTHED COPY TO THE TARGET CRITIC NETWORKS
        self._update_target_networks()