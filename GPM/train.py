import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import datetime
import warnings

from environment import TradingEnvironment
from agent import RLAgent

warnings.filterwarnings("ignore", category=RuntimeWarning)
stock = '2303'
gamma = 0.0
def write_results(episode, train_return, val_return):
    with open('results/results_'+stock+'_'+str(gamma)+'.txt', 'a') as f:
        if episode == 0:
            f.write("Episode,Training_Return,Validation_Return\n")
        f.write(f"{episode},{train_return:.6f},{val_return:.6f}\n")

if __name__ == "__main__":
    gpu_id = 1
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Specified GPU not available. Using default GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")

    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    # Load data
    close = pd.read_csv('../data/close_all.csv', index_col='date', parse_dates=True)
    open_p = pd.read_csv('../data/open_all.csv', index_col='date', parse_dates=True)
    predictions = pd.read_csv("../TimesNet/predictions_"+stock+"_20.csv", index_col='date', parse_dates=True)
    predictions.index = predictions.index + pd.Timedelta(days=56)

    df = pd.DataFrame({
        'Close': close[stock],
        'Open': open_p[stock],
        'Predictions5': predictions['pred_5'],
        'Predictions10': predictions['pred_10']
    })

    df = df.astype(np.float64)
    df = df.dropna()
    print("data loaded")
    train_df = df['2013-01-01':'2019-12-31']
    val_df = df['2020-01-01':'2024-06-24']

    num_episodes = 300
    state_dim = 5
    action_dim = 21  # -10 to 10, inclusive
    agent = RLAgent(state_dim, action_dim, learning_rate=0.00001)
    agent.model = agent.model.to(device)
    env_train = TradingEnvironment(train_df, total_period=len(train_df), n=10)
    env_val = TradingEnvironment(val_df, total_period=len(val_df), n=10)
    resume = False
    print('Environment set')

    checkpoint_dir = 'checkpoints_' + stock
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if resume:
        agent.load_checkpoint('checkpoints_'+stock+'/agent_checkpoint_latest.pth')
        print('checkpoint loaded')

    print("Training started.")
    best_val_return = -np.inf

    for episode in tqdm(range(num_episodes)):
        state = env_train.reset()
        done = False
        states = []
        actions = []
        rewards = []
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env_train.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        equity_return = (env_train.final_equity / env_train.initial_equity) - 1
        loss = agent.update(states, actions, rewards, gamma = gamma)

        print(f"Episode {episode}")
        print(f"Training Return: {equity_return:.2%}")
        print(f"Episode Loss: {loss:.4f}")

        # Run validation
        agent.model.eval()
        val_state = env_val.reset()
        val_done = False
        val_actions = []
        while not val_done:
            val_action = agent.get_action(val_state)
            val_actions.append(val_action)
            val_next_state, _, val_done = env_val.step(val_action)
            val_state = val_next_state

        val_equity_return = (env_val.final_equity / env_val.initial_equity) - 1
        print(f"Validation Return: {val_equity_return:.2%}")
        print(f"Validation Actions: Mean = {np.mean(val_actions):.4f}, Std = {np.std(val_actions):.4f}")
        print("--------------------")

        write_results(episode, equity_return, val_equity_return)

        # Save checkpoint
        agent.save_checkpoint(type = latest, path=checkpoint_dir)

        if val_equity_return > best_val_return:
            best_val_return = val_equity_return
            agent.save_checkpoint(type = best, path=checkpoint_dir)

    print("Training completed.")