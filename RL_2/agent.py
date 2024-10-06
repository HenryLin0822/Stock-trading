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
from utils import prepare_data, prepare_allocation_data
from model import Model
from configs import Strategy_Configs, Allocation_Configs
class DLAgent:
    def __init__(self, strategies, stocks, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.stocks = stocks
        self.action_dim = len(stocks)
        self.strategy_model = []
        self.strategy_configs = Strategy_Configs()
        self.allocation_configs = Allocation_Configs()
        for strategy_ind in range((len(strategies))):
            temp_model = Model(self.strategy_configs).double()
            self.strategy_model.append(temp_model)
        self.allocation_model = Model(self.allocation_configs).double()
        self.optimizer = optim.Adam(self.allocation_model.parameters(), lr=learning_rate)
        self.seq_len = 20
        self.pred_len = 1
    def get_strategy(self, state_df,stock_ind):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                X = prepare_data(state_df, self.seq_len, self.pred_len)
                if len(X) == self.seq_len:
                    X = torch.from_numpy(X).to(torch.float64)
                    X = X.unsqueeze(0)
                    strategy_values = self.strategy_model[stock_ind](X, None, None, None)
                    #print(f"Action values in get_action: {action_values}")
                    #print(f"Action values dtype: {action_values.dtype}")
                    return torch.argmax(strategy_values[0]).item()
        return 0  # Default action if conditions are not met
    def get_allocation(self, state_df_total):
        with torch.no_grad():
            X = prepare_allocation_data(state_df_total, self.stocks)
            if X!=None and len(X) == self.seq_len:
                X = torch.from_numpy(X).to(torch.float64)
                X = X.unsqueeze(0)
                allocation_values = self.strategy_model(X, None, None, None)
                return torch.argmax(allocation_values[0]).item()
        return 0
    def update(self, state_df_total, stock_ind, reward):
        reward_tensor = torch.tensor([reward * 100], dtype=torch.float64)
        
        X = prepare_allocation_data(state_df_total, self.stocks)
        if X!=None and len(X) == self.seq_len:
                X = torch.from_numpy(X).to(torch.float64)
                X = X.unsqueeze(0)
                allocation_values = self.strategy_model(X, None, None, None)            
            
            #print(f"Action values shape: {action_values.shape}")
            #print(f"Action values dtype: {action_values.dtype}")
            #print(f"Action: {action}")
            #print(f"Reward: {reward_tensor[0]}")
            
                if stock_ind < allocation_values.size(1):
                    predicted_value = allocation_values[0][stock_ind]
                    #print(f"Predicted value dtype: {predicted_value.dtype}")
                    loss = F.mse_loss(predicted_value.unsqueeze(0), reward_tensor)
                    #print(f"Loss dtype: {loss.dtype}")
                    loss.backward()
                    self.optimizer.step()
                    return loss.item()
                else:
                    print(f"Warning: Action {stock_ind} is out of bounds for action_values of shape {action_values.shape}")
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