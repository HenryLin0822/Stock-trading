import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from utils import prepare_data
from model import Model, Configs

class DLAgent:
    def __init__(self, state_dim, learning_rate=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = Configs()
        self.configs.input_dim = state_dim
        self.model = Model(self.configs).double().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

    def get_action(self, state_df):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            X = prepare_data(state_df, self.configs.seq_len, self.configs.pred_len)
            X = torch.from_numpy(X).unsqueeze(0).to(torch.float64).to(self.device)
            action_value = self.model(X)
            return action_value.item()

    def update(self, state_df_list, optimal_action_list):
        print("Start Updating")
        self.model.train()  # Set the model to training mode
        processed_states = []
        for state_df in state_df_list:
            X = prepare_data(state_df, self.configs.seq_len, self.configs.pred_len)
            processed_states.append(X)
        
        states = torch.tensor(np.array(processed_states), dtype=torch.float64).to(self.device)
        optimal_actions = torch.tensor(optimal_action_list, dtype=torch.float64).to(self.device).unsqueeze(1)
        
        predicted_actions = self.model(states)
        
        # Use MSE loss
        loss = nn.MSELoss()(predicted_actions, optimal_actions)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        print("Finish Updating")

        return loss.item()

    def save_checkpoint(self, episode, path='checkpoints'):
        if not os.path.exists(path):
            os.makedirs(path)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, f'{path}/agent_checkpoint_ep{episode}.pth')
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['episode']