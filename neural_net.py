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

class NeuralNet(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_action,
                 n_hidden_layers=1,
                 hidden_dim=32):
        super(NeuralNet, self).__init__()
        
        M = n_inputs
        self.layers = []
        for _ in range(n_hidden_layers):
            layer = torch.nn.Linear(M, hidden_dim)
            M = hidden_dim
            self.layers.append(layer)
            self.layers.append(torch.nn.ReLU())
            
        self.layers.append(torch.nn.Linear(M, n_action))
        self.layers = torch.nn.Sequential(*self.layers)
        
    def forward(self, X):
        return self.layers(X)
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        
def predict(model, np_states):
    with torch.no_grad():
        inputs = torch.from_numpy(np_states.astype(np.float32))
        output = model(inputs)
        return output.numpy()
    
def train_one_step(model, criterion, optimizer, inputs, targets):
    inputs = torch.from_numpy(inputs.astype(np.float32))
    targets = torch.from_numpy(targets.astype(np.float32))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
