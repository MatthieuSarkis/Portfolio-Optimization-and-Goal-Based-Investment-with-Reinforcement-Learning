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
                 eta1: float, 
                 eta2: float, 
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

        self.actor = Actor(eta2, 
                           input_shape, 
                           layer1_size,
                           layer2_size, 
                           action_space_dimension=action_space_dimension, 
                           name=agent_name+'_actor',
                           max_actions=env.action_space.high)
        
        self.critic_1 = Critic(eta1, 
                               input_shape, 
                               layer1_size,
                               layer2_size, 
                               action_space_dimension=action_space_dimension, 
                               name=agent_name+'_critic1')
        
        self.critic_2 = Critic(eta1, 
                               input_shape, 
                               layer1_size,
                               layer2_size, 
                               action_space_dimension=action_space_dimension, 
                               name=agent_name+'_critic2')
        
        self.value = Value(eta1, 
                           input_shape, 
                           layer1_size,
                           layer2_size, 
                           name=agent_name+'_value')
        
        self.target_value = Value(eta1, 
                                  input_shape, 
                                  layer1_size,
                                  layer2_size, 
                                  name=agent_name+'_target_value')
        
        self._update_target_network(tau=1)
        
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
        
        state, action, reward, new_state, done = self.memory.sample(self.batch_size)
        
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic_1.device)
        done = torch.tensor(done).to(self.critic_1.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.critic_1.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic_1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic_1.device)
        
        # VALUE UPDATE
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0
        
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)        
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)    
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * torch.nn.functional.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True) 
        self.value.optimizer.step()
        
        # POLICY UPDATE
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        
        # CRITIC UPDATE
        q_hat = reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * torch.nn.functional.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * torch.nn.functional.mse_loss(q2_old_policy, q_hat)
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        # EXPONENTIALLY SMOOTHED COPY TO THE TARGET CRITIC NETWORKS
        self._update_target_network()
           
    
class Agent_newSAC():
    def __init__(self, 
                 eta1: float, 
                 eta2: float, 
                 eta3: float,
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
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(action_space_dimension).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=eta3)

        self.actor = Actor(eta2, 
                           input_shape, 
                           layer1_size,
                           layer2_size, 
                           action_space_dimension=action_space_dimension, 
                           name=agent_name+'_actor',
                           max_actions=env.action_space.high)
        
        self.critic_1 = Critic(eta1, 
                               input_shape, 
                               layer1_size,
                               layer2_size, 
                               action_space_dimension=action_space_dimension, 
                               name=agent_name+'_critic1')
        
        self.critic_2 = Critic(eta1, 
                               input_shape, 
                               layer1_size,
                               layer2_size, 
                               action_space_dimension=action_space_dimension, 
                               name=agent_name+'_critic2')
        
        self.target_critic_1 = Critic(eta1, 
                                      input_shape, 
                                      layer1_size,
                                      layer2_size, 
                                      action_space_dimension=action_space_dimension, 
                                      name=agent_name+'_target_critic1')
        
        self.target_critic_2 = Critic(eta1, 
                                      input_shape, 
                                      layer1_size,
                                      layer2_size, 
                                      action_space_dimension=action_space_dimension, 
                                      name=agent_name+'_target_critic2')
        
        self._update_target_networks(tau=1)
        
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
        
    def _update_target_networks(self, 
                              tau: float = None) -> None:
        
        if tau is None:
            tau = self.tau

        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_1_params = self.value.named_parameters()
        critic_2_params = self.value.named_parameters()
        
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
        
    def save_networks(self) -> None:
        
        print(' ... saving models ... ')
        self.actor.save_network_weights()
        self.critic_1.save_network_weights()
        self.critic_2.save_network_weights()
        self.target_critic_1.save_network_weights()
        self.target_critic_2.save_network_weights()
        
    def load_networks(self) -> None:
        
        print(' ... loading models ... ')
        self.actor.load_network_weights()
        self.critic_1.load_network_weights()
        self.critic_2.load_network_weights()
        self.target_critic_1.load_network_weights()
        self.target_critic_2.load_network_weights()
        
    def learn(self) -> None:
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.critic_1.device)
        dones = torch.tensor(dones).to(self.critic_1.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.critic_1.device)
        states = torch.tensor(states, dtype=torch.float).to(self.critic_1.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.critic_1.device)
        
        # CRITIC UPDATE
        actions_, log_probs_ = self.actor.sample_normal(states_, reparameterize=False)
        q1_ = self.critic_1.forward(states_, actions_).view(-1)
        q2_ = self.critic_2.forward(states_, actions_).view(-1)
        min_q_ = torch.min(q1_, q2_) - self.alpha * log_probs_
        q_target = rewards + (1 - dones) * self.gamma * min_q_
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
        actions, log_probs = self.actor.sample_normal(states, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(states, actions)
        q2_new_policy = self.critic_2.forward(states, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = self.alpha * log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
                    
        # TEMPERATURE UPDATE
        log_alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        log_alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # EXPONENTIALLY SMOOTHED COPY TO THE TARGET CRITIC NETWORKS
        self._update_target_networks()
         
    