import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import prepare_data
from model import Model, Configs
import os
class RLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = Configs()
        self.configs.input_dim = state_dim
        self.configs.c_out = action_dim
        self.model = Model(self.configs).double().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

    def get_action(self, state_df):
        self.model.eval()
        with torch.no_grad():
            X = prepare_data(state_df, self.configs.seq_len, self.configs.pred_len)
            X = torch.from_numpy(X).unsqueeze(0).to(torch.float64).to(self.device)
            action_probs = F.softmax(self.model(X), dim=1)
            action = torch.multinomial(action_probs, 1).item()
            return action

    def update(self, states, actions, rewards, gamma):
        self.model.train()
        processed_states = [prepare_data(state, self.configs.seq_len, self.configs.pred_len) for state in states]
        states = torch.tensor(np.array(processed_states), dtype=torch.float64).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float64).to(self.device)

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in rewards.flip(0):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate loss
        action_probs = F.softmax(self.model(states), dim=1)
        log_probs = F.log_softmax(self.model(states), dim=1)
        selected_log_probs = log_probs[range(len(actions)), actions]
        loss = -(selected_log_probs * discounted_rewards).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def save_checkpoint(self, type, path='checkpoints'):
        if not os.path.exists(path):
            os.makedirs(path)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if type == 'best':
            torch.save(checkpoint, f'{path}/agent_best.pth')
        torch.save(checkpoint, f'{path}/agent_latest.pth')
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['episode']