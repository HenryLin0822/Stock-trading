# predict.py

import torch
import pandas as pd
import numpy as np
from model import Model
from utils import prepare_data, make_predictions, evaluate_predictions
from torch.utils.data import DataLoader
from utils import StockDataset
import matplotlib.pyplot as plt
from config import Configs
def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    # Load the validation data
    close = pd.read_csv('../data/close_all.csv', index_col='date', parse_dates=True)
    open_p = pd.read_csv('../data/open_all.csv', index_col='date', parse_dates=True)
    high = pd.read_csv('../data/high_all.csv', index_col='date', parse_dates=True)
    low = pd.read_csv('../data/low_all.csv', index_col='date', parse_dates=True)
    money = pd.read_csv('../data/money_all.csv', index_col='date', parse_dates=True)
    num = pd.read_csv('../data/num_all.csv', index_col='date', parse_dates=True)


    stock = '2330'
    df = pd.DataFrame({
        'close': close[stock],
        'open': open_p[stock],
        'high': high[stock],
        'low': low[stock],
        'money': money[stock],
        'num': num[stock]
    })

    val_df = df['2020-01-01':'2024-07-19']

    # Prepare the data
    seq_len = 40  # 4 weeks
    pred_len = 5  # 1 week
    X_val, y_val, scaler = prepare_data(val_df, seq_len, pred_len)

    val_dataset = StockDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    configs = Configs()

    # Initialize the model
    model = Model(configs)

    # Load the trained model
    checkpoint_path = 'checkpoints/checkpoint_epoch_3.pth'  # Update this path to your best model checkpoint
    model = load_model(model, checkpoint_path, device)
    model.to(device)
    model.eval()
    
    # Make predictions
    val_predictions = make_predictions(model, val_loader, device)
    print(f"Debug: val_predictions shape: {val_predictions.shape}")
    print(f"Debug: y_val shape: {y_val.shape}")

    # Evaluate predictions
    mse, inv_predictions, inv_y_true, prediction_dates = evaluate_predictions(val_predictions, y_val, scaler, val_df)
    print(f'Validation MSE: {mse:.4f}')
    #print(inv_predictions)
    # Create a DataFrame with the predictions and dates
    predictions_df = pd.DataFrame(inv_predictions, columns=[f'pred_{i+1}' for i in range(inv_predictions.shape[1])])
    predictions_df['date'] = prediction_dates[:len(predictions_df)]
    predictions_df.set_index('date', inplace=True)
    print(predictions_df)
    # Save predictions to CSV
    predictions_df.to_csv('predictions.csv')
    print("Predictions saved to predictions.csv")

    # Print sample predictions
    # Plot the first 50 predictions vs actual values
    plt.figure(figsize=(15, 6))
    #plt.plot(val_df.index, val_df['close'], label='Actual Close Prices')
    #plt.plot(predictions_df.index, predictions_df['pred_1'], label='Predicted Close Prices')
    plt.plot(inv_predictions[:, 0], label='Predicted')
    plt.plot(inv_y_true[:, 0], label='Actual')
    plt.legend()
    plt.title('Predicted vs Actual Close Prices')
    plt.savefig('predictions_vs_actual.png')
    plt.close()

    
    
if __name__ == "__main__":
    main()