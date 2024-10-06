# utils.py

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
class StockDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def prepare_data(df, seq_len, pred_len):
    data = df.values
    #print(df)
    #print(data)
    X = []
    for i in range(len(df)):
        X.append(np.float64(data[i]))
    return np.array(X)

def prepare_allocation_data(df_total,stocks):
    new_df = pd.DataFrame()
    for i in range(len(stocks)):
        for column_name,column_data in df_total[i].items():
            new_df[column_name+str(i)] = column_data
    #print(new_df)
    data = new_df.values
    X = []
    for i in range(len(new_df)):
        X.append(np.float64(data[i]))
    return np.array(X)
        
def evaluate_predictions(predictions, y_true, scaler, val_df):
    #print(f"Debug: predictions shape: {predictions.shape}")
    #print(f"Debug: y_true shape: {y_true.shape}")
    #print(f"Debug: val_df shape: {val_df.shape}")
    # Ensure predictions and y_true have the same shape
    assert predictions.shape == y_true.shape, "Predictions and y_true shapes do not match"

    #print(f"Debug: After adjustment - predictions shape: {predictions.shape}")
    #print(f"Debug: After adjustment - y_true shape: {y_true.shape}")

    # Prepare dummy array for inverse transform
    dummy = np.zeros((len(predictions) * predictions.shape[1], scaler.n_features_in_))
    
    # Put predictions and y_true into the first column (assuming 'close' is the first feature)
    dummy_pred = dummy.copy()
    dummy_pred[:, 0] = predictions.reshape(-1)
    
    dummy_true = dummy.copy()
    dummy_true[:, 0] = y_true.reshape(-1)
    
    # Inverse transform
    inv_predictions = scaler.inverse_transform(dummy_pred)[:, 0].reshape(predictions.shape)
    inv_y_true = scaler.inverse_transform(dummy_true)[:, 0].reshape(y_true.shape)
    
    # Calculate MSE
    mse = np.mean((inv_predictions - inv_y_true) ** 2)
    
    # Get the dates for the predictions
    prediction_dates = val_df.index[len(val_df) - len(predictions) * predictions.shape[1]:]
    
    return mse, inv_predictions, inv_y_true, prediction_dates

def make_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, _ in tqdm(dataloader):
            X = X.to(device).float()  # Ensure dtype is float32
            outputs = model(X, None, None, None)
            #print(f"Debug: outputs shape: {outputs.shape}")
            outputs = outputs[:,:,0]  # Predicting 'close' prices
            #print(outputs)
            predictions.append(outputs.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    #print(predictions)
    #print(predictions.reshape(-1,5))
    #print(f"Debug: make_predictions output shape: {predictions.shape}")
    return predictions.reshape(-1, 5)  # Ensure the output shape is (N, 5)



def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
