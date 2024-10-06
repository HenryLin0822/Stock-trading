from strategies import *
from environment import *
from agent import *
import torch.cuda
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
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Print the device being used
print(f"Using device: {device}")

close = pd.read_csv('../data/close_all.csv', index_col='date', parse_dates=True)
open_p = pd.read_csv('../data/open_all.csv', index_col='date', parse_dates=True)
high = pd.read_csv('../data/high_all.csv', index_col='date', parse_dates=True)
low = pd.read_csv('../data/low_all.csv', index_col='date', parse_dates=True)
volume = pd.read_csv('../data/num_all.csv', index_col='date', parse_dates=True)
money = pd.read_csv('../data/money_all.csv', index_col='date', parse_dates=True)
print('Data loaded')
stock = '2330'
df = pd.DataFrame({
    'Close': close[stock],
    'Open': open_p[stock],
    'High': high[stock],
    'Low': low[stock],
    'Volume': volume[stock],
    'Money': money[stock]
})
df['RSI'] = calculate_rsi(df['Close'])
df['MACD'], df['Signal'] = calculate_macd(df['Close'])
df['K'], df['D'] = calculate_kd(df['High'], df['Low'], df['Close'])
df = df.astype(np.float64)
df = df.dropna()
df_train = df['2008-01-01':'2019-11-01']
df_val = df['2019-11-02':]
#print(df_train)
#print(df_val)

strategies = [RSI, SmaCross, MACD, KD, BuyAndHold, MeanReversion]
# Main training loop

num_episodes = 1000
state_dim = 11  # Adjusted for new features
action_dim = len(strategies)
agent = DLAgent(state_dim, action_dim, learning_rate=0.001, epsilon=1.0)
agent.model = agent.model.to(device)
'''print("Checking model parameter types after initialization:")
for name, param in agent.model.named_parameters():
    print(f"Parameter {name}: dtype = {param.dtype}")'''
env_train = TradingEnvironment(df_train, strategies)
env_val = TradingEnvironment(df_val, strategies)
print('Environment set')
# Checkpoint settings
checkpoint_dir = 'checkpoints_'+stock
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

resume_training = False  # Set to True if you want to resume from a checkpoint
resume_checkpoint = 'checkpoints_'+stock+'/agent_checkpoint_latest.pth'  # Specify the checkpoint to resume from

start_episode = 0
if resume_training:
    if os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, weights_only=True)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode']
        print(f"Resumed training from episode {start_episode}")
    else:
        print(f"Checkpoint {resume_checkpoint} not found. Starting from scratch.")
training = False

print("Training started.")
print("stock:",stock)
best_val_reward = -1000
for episode in tqdm(range(start_episode, num_episodes)):
    state = env_train.reset()
    initial_cash = env_train.current_cash
    done = False
    episode_reward = 0
    episode_loss = 0
    steps = 0
    if training:
        agent.model.train()
        while not done:
            state_df = env_train._get_state()
            #print(state_df)
            action = agent.get_action(state_df,True)
            next_state, reward, done = env_train.step(action)
            reward = float(reward)
            if reward is None:
                print(f"Warning: Received None reward in episode {episode}, step {steps}")
                reward = 0
            episode_reward += reward
            #print(state_df)
            loss = agent.update(state_df, action, reward)
            if loss is not None:
                episode_loss += loss
            else:
                #print(f"Warning: Received None loss in episode {episode}, step {steps}")
                pass
            state = next_state
            steps += 1

        agent.decay_epsilon()
        total_reward = (env_train.current_cash - initial_cash) / initial_cash

        print(f"Episode {episode}")
        print(f"Training Reward: {total_reward:.2%}, Final Cash: ${env_train.current_cash:.2f}")
        print(f"Episode Steps: {steps}, Avg Reward: {episode_reward/max(1,steps):.4f}, Avg Loss: {episode_loss/max(1,steps):.4f}")
        print(f"Epsilon: {agent.epsilon:.4f}")

    # Run validation
    agent.model.eval()
    val_state = env_val.reset()
    val_initial_cash = env_val.current_cash
    val_done = False
    val_steps = 0
    val_episode_reward = 0

    while not val_done:
        #print(val_state)
        val_action = agent.get_action(val_state, False)
        val_action = 4
        print(val_action)
        val_next_state, val_reward, val_done = env_val.step(val_action)
        if val_reward is None:
            print(f"Warning: Received None reward in validation, step {val_steps}")
            val_reward = 0
        val_state = val_next_state
        val_steps += 1
        val_episode_reward += val_reward

    val_total_reward = (env_val.current_cash - val_initial_cash) / val_initial_cash
    print(f"Validation Reward: {val_total_reward:.2%}, Final Cash: ${env_val.current_cash:.2f}")
    print(f"Validation Steps: {val_steps}, Avg Reward: {val_episode_reward/max(1,val_steps):.4f}")
    print("--------------------")
    if training:
        write_reward(episode, total_reward, val_total_reward)
    

print("Training completed.")