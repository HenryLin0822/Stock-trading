import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import datetime
from backtesting.lib import crossover
from ta.momentum import RSIIndicator
from collections import defaultdict
import ta
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover

class RSI(Strategy):
    rsi_window = 14  # Define rsi_window as a class attribute

    def init(self):
        close_series = pd.Series(self.data.Close)
        self.rsi = self.I(lambda x: RSIIndicator(x, self.rsi_window).rsi(), close_series)


    def next(self):
        if crossover(self.rsi, 70):  # RSI crosses above 70 (overbought)
            self.position.close()
        elif crossover(30, self.rsi):  # RSI crosses below 30 (oversold)
            self.buy()

class MACD_RSI(Strategy):
    def init(self):
        close = pd.Series(self.data.Close)
        rsi_indicator = ta.momentum.RSIIndicator(close, window=14)
        macd_indicator = ta.trend.MACD(close, 26, 12, 9, False)
        self.macd = macd_indicator.macd()
        self.signal = macd_indicator.macd_signal()
        self.rsi = self.I(rsi_indicator.rsi)
        self.macd = self.I(macd_indicator.macd)

    def next(self):
        if crossover(self.macd, self.signal) and self.rsi[-1] < 30:
            self.buy()
        elif crossover(self.signal, self.macd) and self.rsi[-1] > 70:
            self.position.close()

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()

class KD(Strategy):
    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)

        self.k = self.I(stoch.stoch)
        self.d = self.I(stoch.stoch_signal)

    def next(self):
        if crossover(self.k, self.d) and self.k[-1] < 20:
            self.buy()
        elif crossover(self.d, self.k) and self.k[-1] > 80:
            self.position.close()

class MACD(Strategy):
    def init(self):
        close = pd.Series(self.data.Close)
        macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)

        self.macd = self.I(macd_indicator.macd)
        self.signal = self.I(macd_indicator.macd_signal)

    def next(self):
        if crossover(self.macd, self.signal):
            self.buy()
        elif crossover(self.signal, self.macd):
            self.position.close()

class MeanReversion(Strategy):
    def init(self):
        close = pd.Series(self.data.Close)
        self.sma = self.I(ta.trend.SMAIndicator(close, window=20).sma_indicator)
        self.std = self.I(close.rolling(window=20).std)

    def next(self):
        if (self.data.Close[-1] < self.sma[-1] - 2 * self.std[-1]):
            self.buy()
        elif (self.data.Close[-1] > self.sma[-1] + 2 * self.std[-1]):
            self.position.close()

class BuyAndHold(Strategy):
    def init(self):
        pass  # No initialization needed for this simple strategy

    def next(self):
        if not self.position:  # If we don't have a position yet
            self.buy()  # Buy the entire portfolio