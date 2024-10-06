# predict.py

import torch
import pandas as pd
import numpy as np
from iTransformer import Model
from utils import prepare_data, make_predictions, evaluate_predictions
from torch.utils.data import DataLoader
from utils import StockDataset
import matplotlib.pyplot as plt
def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    # Load the validation data
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

    val_df = df['2020-01-01':'2020-12-31']

    # Prepare the data
    seq_len = 20  # 4 weeks
    pred_len = 5  # 1 week
    X_val, y_val, scaler = prepare_data(val_df, seq_len, pred_len)

    val_dataset = StockDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model configuration (make sure this matches your training configuration)
    class Config:
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
        enc_in = 6  # number of input features

    configs = Config()

    # Initialize the model
    model = Model(configs)

    # Load the trained model
    checkpoint_path = 'checkpoints/best_model.pth'  # Update this path to your best model checkpoint
    model = load_model(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    # Make predictions
    val_predictions = make_predictions(model, val_loader, device)

    # Evaluate predictions
    mse, inv_predictions, inv_y_true, prediction_dates = evaluate_predictions(val_predictions, y_val, scaler, val_df)
    print(f'Validation MSE: {mse:.4f}')

    # Create a DataFrame with the predictions and dates
    predictions_df = pd.DataFrame(inv_predictions, columns=[f'pred_{i+1}' for i in range(inv_predictions.shape[1])])
    predictions_df['date'] = prediction_dates[:len(predictions_df)]
    predictions_df.set_index('date', inplace=True)

    # Save predictions to CSV
    predictions_df.to_csv('predictions.csv')
    print("Predictions saved to predictions.csv")

    # Print sample predictions
    # Plot the first 50 predictions vs actual values
    plt.figure(figsize=(15, 6))
    plt.plot(inv_predictions[:, 0], label='Predicted')
    plt.plot(inv_y_true[:, 0], label='Actual')
    plt.legend()
    plt.title('Predicted vs Actual Close Prices')
    plt.savefig('predictions_vs_actual.png')
    plt.close()

    
    
if __name__ == "__main__":
    main()