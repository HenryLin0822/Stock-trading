import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = configs.input_dim
        self.hidden_dim = configs.hidden_dim
        self.num_layers = configs.num_layers
        self.output_dim = configs.c_out
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True).double()
        self.fc1 = nn.Linear(self.hidden_dim, 64).double()
        self.fc2 = nn.Linear(64, self.output_dim).double()
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).double().to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).double().to(x.device)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use the last output of the LSTM
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class Configs:
    def __init__(self):
        self.input_dim = 5  # Open, Close, Predictions5, Predictions10, TimePortion
        self.hidden_dim = 128
        self.num_layers = 2
        self.c_out = 1  # Output dimension (single continuous value)
        self.seq_len = 1  # Set to 1 as prepare_data returns a single time step
        self.pred_len = 1  # Set to 1 as we're predicting a single action