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
import torch
from src.buffer import ReplayBuffer
from src.networks import Actor, Critic, Value
import gym

class Agent():
    def __init__(self, 
                 lr_Q: float, 
                 lr_pi: float, 
                 input_shape: tuple, 
                 tau: float, 
                 env: gym.Env, 
                 agent_name: str, 
                 action_space_dimension: int,
                 gamma: float = 0.99,  
                 size: int = 1000000,
                 layer1_size: int = 256, 
                 layer2_size: int = 256, 
                 batch_size: int = 100, 
                 ) -> None:
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(size, input_shape, action_space_dimension)
        self.batch_size = batch_size
        self.action_space_dimension = action_space_dimension

        self.actor = Actor(lr_pi, 
                           input_shape, 
                           layer1_size,
                           layer2_size, 
                           action_space_dimension=action_space_dimension, 
                           name=agent_name+'_actor',
                           max_actions=env.action_space.high)
        
        self.critic_1 = Critic(lr_Q, 
                               input_shape, 
                               layer1_size,
                               layer2_size, 
                               action_space_dimension=action_space_dimension, 
                               name=agent_name+'_critic1')
        
        self.critic_2 = Critic(lr_Q, 
                               input_shape, 
                               layer1_size,
                               layer2_size, 
                               action_space_dimension=action_space_dimension, 
                               name=agent_name+'_critic2')
        
        self.value = Value(lr_Q, 
                           input_shape, 
                           layer1_size,
                           layer2_size, 
                           name=agent_name+'_value')
        
        self.target_value = Value(lr_Q, 
                                  input_shape, 
                                  layer1_size,
                                  layer2_size, 
                                  name=agent_name+'_target_value')
        
        self._update_target_network(tau=1)
        
        if torch.cuda.device_count() > 1:
            print("We are using", torch.cuda.device_count(), "GPUs.")
            self = torch.nn.DataParallel(self)
            
        else:
            print("No multiple GPUs available...")
            
        self.to(self.critic_1.device)
        
    def choose_action(self, 
                      observation: np.array) -> np.array:
        
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        
        return actions.cpu().detach().numpy()[0]
    
    def remember(self, 
                 state: np.array, 
                 action: np.array, 
                 reward: float, 
                 new_state: np.array, 
                 done: bool) -> None:
        
        self.memory.push(state, action, reward, new_state, done)
        
    def _update_target_network(self, 
                               tau: float = None) -> None:
        
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)
        
        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_state_dict[name].clone()
            
        self.target_value.load_state_dict(value_state_dict)
        
    def save_networks(self) -> None:
        
        print(' ... saving models ... ')
        
        self.actor.save_network_weights()
        self.value.save_network_weights()
        self.target_value.save_network_weights()
        self.critic_1.save_network_weights()
        self.critic_2.save_network_weights()
        
    def load_networks(self) -> None:
        
        print(' ... loading models ... ')
        
        self.actor.load_network_weights()
        self.value.load_network_weights()
        self.target_value.load_network_weights()
        self.critic_1.load_network_weights()
        self.critic_2.load_network_weights()
        
    def learn(self) -> None:
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float).to(self.critic_1.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.critic_1.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.critic_1.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.critic_1.device)
        dones = torch.tensor(dones).to(self.critic_1.device)
        
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
        q_target = rewards+ self.gamma * value_
        
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
           
class Agent_AutomaticTemperature():
    
    def __init__(self, 
                 lr_Q: float, 
                 lr_pi: float, 
                 lr_alpha: float,
                 input_shape: tuple, 
                 tau: float, 
                 env: gym.Env, 
                 agent_name: str, 
                 action_space_dimension: int, 
                 gamma: float = 0.99, 
                 size: int = 1000000,
                 layer1_size: int = 256, 
                 layer2_size: int = 256, 
                 batch_size: int = 100, 
                 alpha: float = 1.0
                 ) -> None:
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(size, input_shape, action_space_dimension)
        self.batch_size = batch_size
        self.action_space_dimension = action_space_dimension

        self.actor = Actor(lr_pi, 
                           input_shape, 
                           layer1_size,
                           layer2_size, 
                           n_actions=action_space_dimension, 
                           name=agent_name+'_actor',
                           max_actions=env.action_space.high)
        
        self.critic_1 = Critic(lr_Q, 
                               input_shape, 
                               layer1_size,
                               layer2_size, 
                               n_actions=action_space_dimension, 
                               name=agent_name+'_critic1')
        
        self.critic_2 = Critic(lr_Q, 
                               input_shape, 
                               layer1_size,
                               layer2_size, 
                               n_actions=action_space_dimension, 
                               name=agent_name+'_critic2')
        
        self.target_critic_1 = Critic(lr_Q, 
                                      input_shape, 
                                      layer1_size,
                                      layer2_size, 
                                      n_actions=action_space_dimension, 
                                      name=agent_name+'_target_critic1')
        
        self.target_critic_2 = Critic(lr_Q, 
                                      input_shape, 
                                      layer1_size,
                                      layer2_size, 
                                      n_actions=action_space_dimension, 
                                      name=agent_name+'_target_critic2')
        
        self.actor.apply(self._initialize_weights)
        self.critic_1.apply(self._initialize_weights)
        self.critic_2.apply(self._initialize_weights)
        
        self._update_target_networks(tau=1)
        
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(action_space_dimension).to(self.critic_1.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True).to(self.critic_1.device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
        if torch.cuda.device_count() > 1:
            print("We are using", torch.cuda.device_count(), "GPUs.")
            self = torch.nn.DataParallel(self)
            
        else:
            print("No multiple GPUs available...")
            
        self.to(self.critic_1.device)
        
    def _initialize_weights(net: torch.nn.Module) -> None:
        
        if type(net) == torch.nn.Linear:
            torch.nn.init.xavier_uniform(net.weight)
            net.bias.data.fill_(1e-2)
        
    def choose_action(self, 
                      observation: np.array) -> np.array:
        
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        
        return actions.cpu().detach().numpy()[0]
    
    def remember(self, 
                 state: np.array, 
                 action: np.array, 
                 reward: float, 
                 new_state: np.array, 
                 done: bool) -> None:
        
        self.memory.push(state, action, reward, new_state, done)
        
    def learn(self) -> None:
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
               
        states = torch.tensor(states, dtype=torch.float).to(self.critic_1.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.critic_1.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.critic_1.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.critic_1.device)
        dones = torch.tensor(dones).to(self.critic_1.device)
        
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
        critic_value = critic_value
        
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
     
    def _update_target_networks(self, 
                                tau: float = None) -> None:
    
        if tau is None:
            tau = self.tau

        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        
        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + (1 - tau) * target_critic_1_state_dict[name].clone()
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + (1 - tau) * target_critic_2_state_dict[name].clone()
            
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        
    def save_models(self) -> None:
        
        print(' ... saving models ... ')
        
        self.actor.save_network_weights()
        self.critic_1.save_network_weights()
        self.critic_2.save_network_weights()
        self.target_critic_1.save_network_weights()
        self.target_critic_2.save_network_weights()
        
    def load_models(self) -> None:
        
        print(' ... loading models ... ')
        
        self.actor.load_network_weights()
        self.critic_1.load_network_weights()
        self.critic_2.load_network_weights()
        self.target_critic_1.load_network_weights()
        self.target_critic_2.load_network_weights()