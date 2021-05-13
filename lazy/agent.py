import numpy as np
from replay_buffer import ReplayBuffer
from neural_net import NeuralNet, predict, train_one_step
import torch.nn as nn
import torch

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, size=500)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.brain = NeuralNet(state_size, action_size) 
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.brain.parameters())
        
    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = predict(self.brain, state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        if self.memory.size < batch_size:
            return
        
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']
        
        target = rewards + (1 - done) * self.gamma * np.max(predict(self.brain, next_states), axis=1)
        target_full = predict(self.brain, states)
        target_full[np.arange(batch_size), actions] = target
        
        train_one_step(self.brain, self.criterion, self.optimizer, states, target_full)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.brain.load_weights(name)
        
    def save(self, name):
        self.brain.save_weights(name)