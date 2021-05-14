import os
import torch
import numpy as np
from buffer import ReplayBuffer
from sac_networks import Actor, Critic, Value

class Agent():
    def __init__(self, 
                 eta2, 
                 eta1, 
                 input_dims, 
                 tau, 
                 env, 
                 env_id, 
                 gamma=0.99, 
                 action_space_dim=2, 
                 size=1000000,
                 layer1_size=256, 
                 layer2_size=256, 
                 batch_size=100, 
                 temperature=2):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(size, input_dims, action_space_dim)
        self.batch_size = batch_size
        self.action_space_dim = action_space_dim

        self.actor = Actor(eta2, 
                           input_dims, 
                           layer1_size,
                           layer2_size, 
                           action_space_dim=action_space_dim, 
                           name=env_id+'_actor',
                           max_actions=env.action_space.high)
        
        self.critic_1 = Critic(eta1, 
                               input_dims, 
                               layer1_size,
                               layer2_size, 
                               action_space_dim=action_space_dim, 
                               name=env_id+'_critic1')
        
        self.critic_2 = Critic(eta1, 
                               input_dims, 
                               layer1_size,
                               layer2_size, 
                               action_space_dim=action_space_dim, 
                               name=env_id+'_critic2')
        
        self.value = Value(eta1, 
                           input_dims, 
                           layer1_size,
                           layer2_size, 
                           name=env_id+'_value')
        
        self.target_value = Value(eta1, 
                                  input_dims, 
                                  layer1_size,
                                  layer2_size, 
                                  name=env_id+'_target_value')
        
        self.temperature = temperature
        self.update_target_network(tau=1)
        
    def choose_action(self, 
                      observation):
        
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        
        return actions.cpu().detach().numpy()[0]
    
    def remember(self, 
                 state, 
                 action, 
                 reward, 
                 new_state, 
                 done):
        
        self.memory.store_memory(state, action, reward, new_state, done)
        
    def update_target_network(self, 
                              tau=None):
        
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)
        
        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_state_dict[name].clone()
            
        self.target_value.load_state_dict(value_state_dict)
        
    def save_models(self):
        
        print(' ... saving models ... ')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        
    def load_models(self):
        
        print(' ... loading models ... ')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        
    def learn(self):
        
        if self.memory.pointer < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_memories(self.batch_size)
        
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic_1.device)
        done = torch.tensor(done).to(self.critic_1.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.critic_1.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic_1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic_1.device)
        
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
        value_loss.backward(retain_graph=True) # because there is a lot of coupling between the various loss functions
        self.value.optimizer.step()
        
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
        
        q_hat = self.temperature * reward + self.gamma * value_
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
        
        self.update_target_network()
         