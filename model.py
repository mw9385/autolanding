import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, Linear

class Linear_Model(nn.Module):
    def __init__(self, state_dim, time_step, n_hidden, n_output):        
        super(Linear_Model, self).__init__()

        self.l1 = Linear(state_dim * time_step, n_hidden)
        self.l2 = Linear(n_hidden, n_hidden)
        self.l3 = Linear(n_hidden, n_output)
                
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)                
        return x

class RNN_Model(nn.Module):
    def __init__(self, state_dim, n_hidden, n_lstm, n_output):
        super(RNN_Model, self).__init__()        
        self.lstm = LSTM(input_size = state_dim, hidden_size = n_lstm, num_layers = 1, batch_first = True) 
        self.l1 = Linear(n_lstm, n_hidden)
        self.l2 = Linear(n_hidden, n_output)

    def forward(self, x, hidden=None):
        x, x_hidden = self.lstm(x, hidden)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x, x_hidden