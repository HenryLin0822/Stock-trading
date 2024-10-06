import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_data(df, seq_len, pred_len):
    data = df.values

    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len, 0])  # Predicting 'close' prices
    return np.array(X), np.array(y)


seq_len = 20  # 4 weeks
pred_len = 5  # 1 week

##randomly assign number
df = pd.DataFrame({
    'close': np.array([i for i in range(1,50)]),
    'open': np.array([i for i in range(1,50)]),
    'high': np.array([i for i in range(1,50)]),
    'low': np.array([i for i in range(1,50)]),
    
})
print(df)
X,y  = prepare_data(df, seq_len, pred_len)
print(X)
print(y)