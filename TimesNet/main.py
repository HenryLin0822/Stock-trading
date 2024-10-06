import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model import Model
from utils import *
from config import Configs
import os
from tqdm import tqdm
# Hyperparameters
batch_size = 32
epochs = 100
learning_rate = 0.001

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device).float(), y.to(device).float()  # Ensure dtype is float32
        optimizer.zero_grad()
        outputs = model(X, None, None, None)
        #print(f"Debug: outputs shape: {outputs.shape}")
        #print(f"Debug: y shape: {y.shape}")
        loss = criterion(outputs[:,:,0], y)  # Predicting 'close' prices
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device).float(), y.to(device).float()  # Ensure dtype is float32
            outputs = model(X, None, None, None)
            loss = criterion(outputs[:,:,0], y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = 0

    with open('loss.txt', 'w') as f:
        f.write('Epoch,Train Loss,Val Loss\n')
    print("Training...")
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print("evaluating")
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        with open('loss.txt', 'a') as f:
            f.write(f'{epoch+1},{train_loss:.4f},{val_loss:.4f}\n')
        
        # Save checkpoint
        checkpoint_filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(model, optimizer, epoch+1, checkpoint_filename)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_filename = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch+1, best_model_filename)

    # Load the best model for evaluation
    best_epoch = load_checkpoint(model, optimizer, best_model_filename, device)
    print(f'Loaded best model from epoch {best_epoch}')

    return best_epoch

if __name__ == "__main__":
    close = pd.read_csv('../data/close_all.csv', index_col='date', parse_dates=True)
    open_p = pd.read_csv('../data/open_all.csv', index_col='date', parse_dates=True)
    high = pd.read_csv('../data/high_all.csv', index_col='date', parse_dates=True)
    low = pd.read_csv('../data/low_all.csv', index_col='date', parse_dates=True)
    money = pd.read_csv('../data/money_all.csv', index_col='date', parse_dates=True)
    num = pd.read_csv('../data/num_all.csv', index_col='date', parse_dates=True)
    foreign_buy = pd.read_csv('../data/foreign_buy.csv', index_col='date', parse_dates=True)
    foreign_sell = pd.read_csv('../data/foreign_sell.csv', index_col='date', parse_dates=True)
    sitc_buy = pd.read_csv('../data/sitc_buy.csv', index_col='date', parse_dates=True)
    sitc_sell = pd.read_csv('../data/sitc_sell.csv', index_col='date', parse_dates=True)

    #stock = '2330'
    #do not run on local
    df = pd.DataFrame({
        'close': close[stock],
        'open': open_p[stock],
        'high': high[stock],
        'low': low[stock],
        'money': money[stock],
        'num': num[stock],
        'foreign_buy': foreign_buy[stock],
        'foreign_sell': foreign_sell[stock],
        'sitc_buy': sitc_buy[stock],
        'sitc_sell': sitc_sell[stock]
    })
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['Signal'] = calculate_macd(df['close'])
    df['K'], df['D'] = calculate_kd(df['high'], df['low'], df['close'])
    df = df.astype(np.float64)
    df = df.dropna()
    print("data loaded")
    train_df = df['2008-01-01':'2019-12-31']
    val_df = df['2020-01-01':'2024-07-19']

    seq_len = 40  # 4 weeks
    pred_len = 5  # 1 week

    X_train, y_train, scaler = prepare_data(train_df, seq_len, pred_len)
    X_val, y_val, _ = prepare_data(val_df, seq_len, pred_len)

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print("Data prepared")
    configs = Configs()
    model = Model(configs)
    print("Model initialized")
    #model.load_state_dict(torch.load('checkpoints/checkpoint_epoch_3.pth')['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    best_epoch = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    print(f"Debug: X_val shape: {X_val.shape}")
    print(f"Debug: y_val shape: {y_val.shape}")
    print(f"Debug: val_df shape: {val_df.shape}")

    