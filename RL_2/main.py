from strategies import *
from environment import TradingEnvironment
from agent import DLAgent

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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def write_reward(episode, train_reward, val_reward):
    with open('reward.txt', 'a') as f:
        if episode == 0:
            f.write("Episode,Training_Reward,Validation_Reward\n")
        f.write(f"{episode},{train_reward:.6f},{val_reward:.6f}\n")
# Load data
close = pd.read_csv('../data/close_all.csv', index_col='date', parse_dates=True)
open_p = pd.read_csv('../data/open_all.csv', index_col='date', parse_dates=True)
high = pd.read_csv('../data/high_all.csv', index_col='date', parse_dates=True)
low = pd.read_csv('../data/low_all.csv', index_col='date', parse_dates=True)
volume = pd.read_csv('../data/num_all.csv', index_col='date', parse_dates=True)
money = pd.read_csv('../data/money_all.csv', index_col='date', parse_dates=True)
print('Data loaded')
stocks = ['2330','2454','2308']
df_total_train = []
df_total_val = []
for i in range(len(stocks)):
    df = pd.DataFrame({
        'Close': close[stocks[i]],
        'Open': open_p[stocks[i]],
        'High': high[stocks[i]],
        'Low': low[stocks[i]],
        'Volume': volume[stocks[i]],
        'Money': money[stocks[i]]
    })
    df = df.astype(np.float64)
    df_train = df['2008-01-01':'2020-01-01']
    df_val = df['2020-01-01':]
    df_total_train.append(df_train)
    df_total_val.append(df_val)


strategies = [RSI, SmaCross, MACD, KD, BuyAndHold]
# Main training loop

num_episodes = 1000
state_dim = 9  # Adjusted for new features
action_dim = len(stocks)
agent = DLAgent(strategies, stocks, learning_rate=0.001, epsilon=1.0)
'''print("Checking model parameter types after initialization:")
for name, param in agent.model.named_parameters():
    print(f"Parameter {name}: dtype = {param.dtype}")'''
env_train = TradingEnvironment(df_total_train, strategies,stocks)
env_val = TradingEnvironment(df_total_val, strategies,stocks)
print('Environment set')
# Checkpoint settings
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

resume_training = False  # Set to True if you want to resume from a checkpoint
resume_checkpoint = 'checkpoints/agent_checkpoint_latest.pth'  # Specify the checkpoint to resume from

start_episode = 0
'''if resume_training:
    if os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode']
        print(f"Resumed training from episode {start_episode}")
    else:
        print(f"Checkpoint {resume_checkpoint} not found. Starting from scratch.")'''

print("Training started.")
for episode in tqdm(range(start_episode, num_episodes)):
    state = env_train.reset()
    initial_cash = env_train.current_cash
    done = False
    episode_reward = 0
    episode_loss = 0
    steps = 0

    while not done:
        state_df = env_train._get_state()
        #print(state_df)
        stock_ind = agent.get_allocation(state_df)
        stock = stocks[stock_ind]
        strategy = agent.get_strategy(state_df[stock_ind],stock_ind)
        next_state, reward, done = env_train.step(stock_ind, strategy)
        reward = float(reward)
        if reward is None:
            print(f"Warning: Received None reward in episode {episode}, step {steps}")
            reward = 0
        episode_reward += reward
        loss = agent.update(state_df, stock_ind, reward)
        if loss is not None:
            episode_loss += loss
        else:
            #print(f"Warning: Received None loss in episode {episode}, step {steps}")
            pass
        state = next_state
        steps += 1

    agent.decay_epsilon()
    total_reward = (env_train.current_cash - initial_cash) / initial_cash

    if episode % 1 == 0:
        print(f"Episode {episode}")
        print(f"Training Reward: {total_reward:.2%}, Final Cash: ${env_train.current_cash:.2f}")
        print(f"Episode Steps: {steps}, Avg Reward: {episode_reward/max(1,steps):.4f}, Avg Loss: {episode_loss/max(1,steps):.4f}")
        print(f"Epsilon: {agent.epsilon:.4f}")

        # Run validation
        val_state = env_val.reset()
        val_initial_cash = env_val.current_cash
        val_done = False
        val_steps = 0
        val_episode_reward = 0

        while not val_done:
            state_df = env_val._get_state()
            stock_ind = agent.get_allocation(state_df)
            strategy = agent.get_strategy(state_df[stock_ind],stock_ind)
            next_state, val_reward, val_done = env_val.step(stock_ind, strategy)
            if val_reward is None:
                print(f"Warning: Received None reward in validation, step {val_steps}")
                val_reward = 0
            val_state = next_state
            val_steps += 1
            val_episode_reward += val_reward

        val_total_reward = (env_val.current_cash - val_initial_cash) / val_initial_cash
        print(f"Validation Reward: {val_total_reward:.2%}, Final Cash: ${env_val.current_cash:.2f}")
        print(f"Validation Steps: {val_steps}, Avg Reward: {val_episode_reward/max(1,val_steps):.4f}")
        print("--------------------")
        write_reward(episode, total_reward, val_total_reward)
        # Save checkpoint every 10 episodes
        checkpoint = {
            'episode': episode,
            'model_state_dict': agent.allocation_model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        }
        torch.save(checkpoint, f'{checkpoint_dir}/agent_checkpoint_latest.pth')
        print(f"Checkpoint saved at episode {episode}")

print("Training completed.")