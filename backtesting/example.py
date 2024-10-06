import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import datetime
import pandas_ta as ta

from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG
from ta.momentum import RSIIndicator

import torch
import torch.nn as nn
import os
from tqdm import tqdm


close = pd.read_csv('../data/close_data.csv', index_col = 0, parse_dates = True)
open = pd.read_csv('../data/open_data.csv', index_col = 0, parse_dates = True)
high = pd.read_csv('../data/high_data.csv', index_col = 0, parse_dates = True)
low = pd.read_csv('../data/low_data.csv', index_col = 0, parse_dates = True)


backtesting_df = pd.DataFrame({
    'Close': close['0050'],
    'Open': open['0050'],
    'High': high['0050'],
    'Low': low['0050'],

})
print(backtesting_df)

# Define the strategy class
class Close(Strategy):
    rsi_window = 14  # Define rsi_window as a class attribute

    def init(self):
        close_series = pd.Series(self.data.Close)
        self.rsi = self.I(lambda x: RSIIndicator(x, self.rsi_window).rsi(), close_series)

    def next(self):
        if crossover(self.rsi, 70):  # RSI crosses above 70 (overbought)
            self.sell()
        elif crossover(30, self.rsi):  # RSI crosses below 30 (oversold)
            self.buy()


bt = Backtest( backtesting_df,Close, cash=10000, commission=.002,
             exclusive_orders=True)
bt.run()
bt.plot()