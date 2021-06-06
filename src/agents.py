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
from src.environment import Environment
from src.networks import Actor, Critic, Value, Distributional_Critic
  
   
class Agent():
    """Abstract Agent class to be inherited by the various SAC agents 
    """
    
    def __init__(self,
                 lr_Q: float, 
                 lr_pi: float, 
                 input_shape: Tuple, 
                 tau: float, 
                 env: gym.Env, 
                 agent_name: str, 
                 gamma: float = 0.99, 
                 size: int = 1000000,
                 layer_size: int = 256, 
                 batch_size: int = 256,
                 delay: int = 1,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method of the Agent class.
        
        Args:
            lr_Q (float): learning rate for critic nets
            lr_pi (float): learning rate for policy nets
            input_shape (Tuple): shape of the input data (state, or state and action)
            tau (float): linear interpolation parameter for the smooth copy to the target nets
            env (gym.Env): environment in which the agent evolves
            agent_name (str): name of the agent
            gamma (float): discout factor for the rewards
            size (int): maximal size of the replay buffer
            layer_size (int): number of neurons in the layers of the neural nets
            batch_size (int): size of the batches sampled from the replay buffer in the learning process
            delay (int): number of steps between each update of the policy, temperature and target nets
            device (str): cpu or gpu
            
        Returns:
            no value
        """
        
        self.gamma = gamma
        self.tau = tau
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.lr_Q = lr_Q
        self.lr_pi = lr_pi
        self.env = env
        self.action_space_dimension = env.action_space.shape[0]
        self.memory = ReplayBuffer(size, self.input_shape, self.action_space_dimension)
        self.agent_name = agent_name
        self.layer_size = layer_size
        self.delay = delay
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
        """Store some observation in the replay buffer.
        
        Args:
            state (np.array): observation of the environment state 
            action (np.array): action chosen in that state
            reward (float): reward obtained for taking that action
            new_state (np.array): state in which the environment lands
            done (bool): whether one has reached the horizon or not
        
        Returns:
            no value
        """
        
        self.memory.push(state, action, reward, new_state, done)
            
    @staticmethod   
    def _initialize_weights(net: torch.nn.Module) -> None:
        """Xavier initialization of the weights in the Linear layers of a torch.nn.Module object.
        
        Args:
            net (torch.nn.Module): neural net whose weights are to be initialized
            
        Returns:
            no value
        """
        
        if type(net) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(net.weight)
            net.bias.data.fill_(1e-2)
    
    def _update_target_networks(self, 
                                tau: float = None,
                                ) -> None:
        """Copy the weights of the nets to their respective target net.
        
        Args:
            tau (float): linear interpolation parameter for the smooth copy to the target nets
            
        Returns:
            no value
        """
    
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
        """Choose an action to take given an observation of the state of the environment.
        
        Args:
            observation (np.array): state of the environment
            
        Returns:
            action (np.array) taken in the input state 
        """
           
        state = torch.Tensor([observation]).to(self.device)
        actions, _ = self.actor.sample(state, reparameterize=False)
        
        return actions.cpu().detach().numpy()[0]

    def save_networks(self) -> None:
        """Save checkpoint for the weights of the various nets, used in training mode.
        """
        
        print(' ... saving networks ... ')    
        for network in self._network_list:
            network.save_network_weights()
        
    def load_networks(self) -> None:
        """Loading checkpoint for the weights of the various nets, used in test mode.
        """
        
        print(' ... loading networks ... ')
        for network in self._network_list:
            network.load_network_weights()
   
class Agent_ManualTemperature(Agent):
    """Soft Actor Critic agent according to https://arxiv.org/abs/1801.01290
    
    Inherits from the abstract Agent class.
    Temperature is a hyperparameter to be tuned manually.
    """
    
    def __init__(self, 
                 *args, 
                 **kwargs,
                 ) -> None:
        """Constructor method for the Agent_ManualTemperature class.
        
        No extra input arguments with respect to the mother class.
        """
        
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
        """Implements one learning step by sampling a batch of data from the replay buffer.
        
        The algorithm is explained in https://arxiv.org/abs/1801.01290
        """
        
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
        
        actions, log_probabilities = self.actor.sample(states, reparameterize=False)
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
        actions, log_probabilities = self.actor.sample(states, reparameterize=True)
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
    """Soft Actor Critic agent as introduced in https://arxiv.org/abs/1812.05905
    
    Inherits from the abstract Agent class.
    The temperature parameter is being automatically updated in the learning process.
    """
    
    def __init__(self, 
                 lr_alpha: float,
                 alpha: float = 1.0,
                 *args,
                 **kwargs,
                 ) -> None:
        """Constructor method of the Agent_AutomaticTemperature class.
        
        Args:
            lr_alpha (float): learning rate for the temperature auto adjustment
            alpha (float): initial value of the temperature parameter
            
        Returns:
            no value
        """
        
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
        
    def learn(self,
              step: int = 0,
              ) -> None:
        """Implements one learning step by sampling a batch of data from the replay buffer.
        
        The algorithm is explained in https://arxiv.org/abs/1812.05905
        """
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
               
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        # CRITIC UPDATE
        actions_, log_probabilities_ = self.actor.sample(states_, reparameterize=False)
        
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
        
        if step % self.delay == 0:
        
            # POLICY UPDATE
            actions, log_probabilities = self.actor.sample(states, reparameterize=True)
            
            q1_ = self.target_critic_1.forward(states, actions)
            q2_ = self.target_critic_2.forward(states, actions)
            
            critic_value = torch.min(q1_, q2_)
            
            actor_loss = self.alpha * log_probabilities - critic_value
            actor_loss = torch.mean(actor_loss.view(-1))
            
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            #print(self.actor.layer1.weight.grad)
            #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            
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
    """Distributional Soft Actor Critic agent as introduced in https://arxiv.org/pdf/2001.02811
    
    Inherits from the abstract Agent class.
    The temperature parameter is being automatically updated in the learning process.
    The critic is now fully considered and learned as a random variable, not only its expectation is being learned.
    """
    
    def __init__(self, 
                 lr_alpha: float,
                 alpha: float = 1.0,
                 *args,
                 **kwargs,
                 ) -> None:
        """Constructor method of the Agent_AutomaticTemperature class.
        
        Args:
            lr_alpha (float): learning rate for the temperature auto adjustment
            alpha (float): initial value of the temperature parameter
            
        Returns:
            no value
        """
        
        super(Distributional_Agent, self).__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.action_space_dimension).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True).to(self.device).detach().requires_grad_(True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
        self.critic = Distributional_Critic(self.lr_Q,
                                            self.input_shape,
                                            self.layer_size,
                                            self.action_space_dimension,
                                            self.agent_name,
                                            log_sigma_min=-0.1,
                                            log_sigma_max=4,
                                            checkpoint_directory=self.agent_name+'_critic',
                                            device=self.device)
        
        self.target_critic = Distributional_Critic(self.lr_Q,
                                                   self.input_shape,
                                                   self.layer_size,
                                                   self.action_space_dimension,
                                                   self.agent_name,
                                                   log_sigma_min=-0.1,
                                                   log_sigma_max=4,
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
        
    def learn(self,
              step: int = 0,
              ) -> None:
        """Implements one learning step by sampling a batch of data from the replay buffer.
        
        The algorithm is explained in https://arxiv.org/pdf/2001.02811
        """
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
               
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        # CRITIC UPDATE
        _, mu, sigma = self.critic.sample(states, actions, reparameterize=False)
        action_, log_probabilities_= self.actor.sample(states_, reparameterize=False)
        q_, _, _ = self.target_critic.sample(states_, action_, reparameterize=False)
       
        target_q = rewards + (1 - dones.int()) * self.gamma * (q_ - self.alpha * log_probabilities_)
        target_q_clipped = mu + torch.clamp(target_q - mu, -3 * sigma.mean().item(), 3 * sigma.mean().item())

        normal = torch.distributions.Normal(mu, sigma)
        
        critic_loss = -normal.log_prob(target_q_clipped).mean()
            
        self.critic.optimizer.zero_grad()  
        critic_loss.backward(retain_graph=True)
        self.critic.optimizer.step()
        
        if step % self.delay == 0:
        
            # POLICY UPDATE
            actions, log_probabilities = self.actor.sample(states, reparameterize=True)
            
            critic_value, _, _ = self.target_critic.sample(states, actions, reparameterize=True)
            
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
            
            
def instanciate_agent(env: Environment, 
                      device: str, 
                      args: tuple,
                      ) -> Tuple[Agent, str]:
    """Instanciate the correct type of agent, according to args.agent_type.
    
    Args:
        env (Environment): trading environment
        device (str): cpu or gpu, to be passed to the agents' constructor
        args (tuple): various arguments to be passed to the agents' constructor, typically received \
                      as command line arguments
    
    Returns:
        Agent instance
        file path for the training of testing plots
    """
        
    if args.agent_type == 'automatic_temperature':
        
        agent_name = 'automatic_temperature'
        figure_file = str(args.n_episodes) + 'episodes_AutoTemperature_' + args.mode + '.png'
        agent = Agent_AutomaticTemperature(lr_Q=args.lr_Q,
                                           lr_pi=args.lr_pi, 
                                           lr_alpha=args.lr_alpha,  
                                           agent_name=agent_name, 
                                           input_shape=env.observation_space.shape, 
                                           tau=args.tau,
                                           env=env, 
                                           size=args.memory_size,
                                           batch_size=args.batch_size, 
                                           layer_size=args.layer_size, 
                                           alpha=args.alpha,
                                           delay=args.delay,
                                           device=device)
    
    elif args.agent_type == 'manual_temperature':
        
        agent_name = 'manual_temperature'
        figure_file = str(args.n_episodes) + 'episodes_ManualTemperature_{}'.format(args.sac_temperature) + args.mode + '.png'
        agent = Agent_ManualTemperature(lr_pi=args.lr_pi, 
                                        lr_Q=args.lr_Q, 
                                        gamma=args.gamma, 
                                        agent_name=agent_name, 
                                        input_shape=env.observation_space.shape, 
                                        tau=args.tau,
                                        env=env, 
                                        size=args.memory_size,
                                        batch_size=args.batch_size, 
                                        layer_size=args.layer_size, 
                                        device=device)
        
    elif args.agent_type == 'distributional':
        
        agent_name = 'distributional'
        figure_file = str(args.n_episodes) + 'episodes_Distributional_' + args.mode + '.png'
        agent = Distributional_Agent(lr_Q=args.lr_Q,
                                    lr_pi=args.lr_pi, 
                                    lr_alpha=args.lr_alpha,  
                                    agent_name=agent_name, 
                                    input_shape=env.observation_space.shape, 
                                    tau=args.tau,
                                    env=env, 
                                    size=args.memory_size,
                                    batch_size=args.batch_size, 
                                    layer_size=args.layer_size, 
                                    alpha=args.alpha,
                                    delay=args.delay,
                                    device=device)
        
    return agent, figure_file
