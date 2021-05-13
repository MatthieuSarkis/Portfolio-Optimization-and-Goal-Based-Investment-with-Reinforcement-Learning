import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_action,
                 n_hidden_layers=1,
                 hidden_dim=32):
        super(NeuralNet, self).__init__()
        
        M = n_inputs
        self.layers = []
        for _ in range(n_hidden_layers):
            layer = nn.Linear(M, hidden_dim)
            M = hidden_dim
            self.layers.append(layer)
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(M, n_action))
        self.layers = nn.Sequential(*self.layers)
        
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
