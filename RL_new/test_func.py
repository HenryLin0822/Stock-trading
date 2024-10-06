import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_kd(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    
    return k, d

# Load and prepare data
close = pd.read_csv('../data/close_all.csv', index_col='date', parse_dates=True)
high = pd.read_csv('../data/high_all.csv', index_col='date', parse_dates=True)
low = pd.read_csv('../data/low_all.csv', index_col='date', parse_dates=True)

stock = '2330'
df = pd.DataFrame({
    'Close': close[stock],
    'High': high[stock],
    'Low': low[stock]
})
df_val = df['2020-01-01':]

# Calculate RSI
df_val['RSI'] = calculate_rsi(df_val['Close'])

# Calculate MACD
df_val['MACD'], df_val['Signal'] = calculate_macd(df_val['Close'])

# Calculate KD
df_val['K'], df_val['D'] = calculate_kd(df_val['High'], df_val['Low'], df_val['Close'])

# Display the results
#print(df_val)

# If you want to see only a specific period:
period_start = 0
period_end = 40
period_data = df_val.iloc[period_start:period_end]
print("\nData for the specified period:")
print(period_data)

