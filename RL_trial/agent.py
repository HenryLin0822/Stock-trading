import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import datetime
from backtesting.lib import crossover
from ta.momentum import RSIIndicator
from collections import defaultdict
import ta
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from utils import *
from model import Model
from config import Configs
class DLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim
        self.configs = Configs()
        self.configs.c_out = action_dim
        self.configs.num_class = action_dim
        self.model = Model(self.configs).double()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_action(self, state_df,training):
        if random.random() < self.epsilon and training:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                X = prepare_data(state_df, self.configs.seq_len, self.configs.pred_len)
                if len(X) == self.configs.seq_len:
                    X = torch.from_numpy(X).to(torch.float64)
                    X = X.unsqueeze(0)
                    action_values = self.model(X, None, None, None)
                    #print(f"Action values in get_action: {action_values}")
                    #print(f"Action values dtype: {action_values.dtype}")
                    return torch.argmax(action_values[0]).item()
        return 0  # Default action if conditions are not met
    
    def update(self, state_df, action, reward):
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.tensor([reward * 100], dtype=torch.float64)
        
        X = prepare_data(state_df, self.configs.seq_len, self.configs.pred_len)
        if len(X) == self.configs.seq_len:
            X = torch.from_numpy(X).to(torch.float64)
            X = X.unsqueeze(0)
            self.optimizer.zero_grad()
            action_values = self.model(X, None, None, None)
            
            #print(f"Action values shape: {action_values.shape}")
            #print(f"Action values dtype: {action_values.dtype}")
            #print(f"Action: {action}")
            #print(f"Reward: {reward_tensor[0]}")
            
            if action < action_values.size(1):
                predicted_value = action_values[0][action]
                #print(f"Predicted value dtype: {predicted_value.dtype}")
                loss = F.mse_loss(predicted_value.unsqueeze(0), reward_tensor)
                #print(f"Loss dtype: {loss.dtype}")
                loss.backward()
                self.optimizer.step()
                return loss.item()
            else:
                print(f"Warning: Action {action} is out of bounds for action_values of shape {action_values.shape}")
                return None

        return None

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, episode, path='checkpoints'):
        if not os.path.exists(path):
            os.makedirs(path)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, f'{path}/agent_checkpoint_ep{episode}.pth')
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        return checkpoint['episode']