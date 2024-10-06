
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from iTransformer import Model  # Import your model definition here
from utils import StockDataset, prepare_data, make_predictions, evaluate_predictions, save_checkpoint, load_checkpoint
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class Configs:
    seq_len = 20
    pred_len = 5
    output_attention = False
    use_norm = True
    d_model = 512
    embed = 'timeF'
    freq = 'h'
    dropout = 0.1
    class_strategy = 'token'
    factor = 5
    n_heads = 8
    d_ff = 512
    e_layers = 2
    activation = 'relu'

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device).float(), y.to(device).float()  # Ensure dtype is float32
        optimizer.zero_grad()
        outputs = model(X, None, None, None)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float()  # Ensure dtype is float32
            outputs = model(X, None, None, None)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = 0

    with open('loss.txt', 'w') as f:
        f.write('Epoch,Train Loss,Val Loss\n')

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
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
    # Load and prepare the data
    close = pd.read_csv('../data/close_data.csv', index_col='date', parse_dates=True)
    open_p = pd.read_csv('../data/open_data.csv', index_col='date', parse_dates=True)
    high = pd.read_csv('../data/high_data.csv', index_col='date', parse_dates=True)
    low = pd.read_csv('../data/low_data.csv', index_col='date', parse_dates=True)
    money = pd.read_csv('../data/money_data.csv', index_col='date', parse_dates=True)
    num = pd.read_csv('../data/num_data.csv', index_col='date', parse_dates=True)
    df = pd.DataFrame({
        'close': close['0050'],
        'open': open_p['0050'],
        'high': high['0050'],
        'low': low['0050'],
        'money': money['0050'],
        'num': num['0050']
    })

    train_df = df['2008-01-01':'2019-12-31']
    val_df = df['2020-01-01':'2020-12-31']

    seq_len = 20  # 4 weeks
    pred_len = 5  # 1 week

    X_train, y_train, scaler = prepare_data(train_df, seq_len, pred_len)
    X_val, y_val, _ = prepare_data(val_df, seq_len, pred_len)

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    configs = Configs()
    model = Model(configs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30

    best_epoch = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    print(f"Debug: X_val shape: {X_val.shape}")
    print(f"Debug: y_val shape: {y_val.shape}")
    print(f"Debug: val_df shape: {val_df.shape}")

    