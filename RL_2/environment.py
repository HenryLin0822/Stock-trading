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

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return np.float64(100 - (100 / (1 + rs.iloc[0])))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return np.float64(macd.iloc[0]), np.float64(signal_line.iloc[0])

class TradingEnvironment:
    def __init__(self, df_total, strategies,stocks, initial_cash=10000, period_length=20):
        self.df_total = df_total
        self.strategies = strategies
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.current_period = 0
        self.period_length = period_length
        self.total_periods = len(df_total[0]) // period_length
        self.stocks = stocks
    def _get_state(self):
        if self.current_period == 0:
            # For the first period, we don't have previous data, so we'll use the first day's data
            period_data =[]
            #print(self.df_total[0])
            for stock_ind in range(len(self.stocks)):
                period_data.append(self.df_total[stock_ind].iloc[0:1])
            features_df_total = []
            
            for stock_ind in range(len(self.stocks)):
                close_prices = period_data[stock_ind]['Close']
                rsi = calculate_rsi(close_prices)
                macd, signal = calculate_macd(close_prices)
                
                features_df = pd.DataFrame({
                    'Open': period_data[stock_ind]['Open'],
                    'High': period_data[stock_ind]['High'],
                    'Low': period_data[stock_ind]['Low'],
                    'Close': period_data[stock_ind]['Close'],
                    'Volume': period_data[stock_ind]['Volume'],
                    'Money': period_data[stock_ind]['Money'],
                    'RSI': rsi,
                    'MACD': macd,
                    'Signal': signal
                })
                
                #print(features_df)
                features_df = features_df.fillna(0)
                #print('features_df in _get_state:')
                #print(features_df)
                features_df_total.append(features_df)
        
            return features_df_total                       
            
        else:
            # Use the previous period's data
            period_start = (self.current_period - 1) * self.period_length
            period_end = self.current_period * self.period_length
            period_data =[]
            for stock_ind in range(len(self.stocks)):
                period_data.append(self.df_total[stock_ind].iloc[period_start:period_end])
            
        
        if len(period_data) == 0:
            return np.zeros(9)
        
        features_df_total = []
        
        for stock_ind in range(len(self.stocks)):
            close_prices = period_data[stock_ind]['Close']
            rsi = calculate_rsi(close_prices)
            macd, signal = calculate_macd(close_prices)
            
            features_df = pd.DataFrame({
                'Open': period_data[stock_ind]['Open'],
                'High': period_data[stock_ind]['High'],
                'Low': period_data[stock_ind]['Low'],
                'Close': period_data[stock_ind]['Close'],
                'Volume': period_data[stock_ind]['Volume'],
                'Money': period_data[stock_ind]['Money'],
                'RSI': rsi,
                'MACD': macd,
                'Signal': signal
            })
            
            #print(features_df)
            features_df = features_df.fillna(0)
            #print('features_df in _get_state:')
            #print(features_df)
            features_df_total.append(features_df)
       
        return features_df_total


    def step(self, stock_ind, strategy):
        if self.current_period >= self.total_periods:
            return self._get_state(), 0, True  # Return done if we've reached the end
        period_start = self.current_period * self.period_length
        df = self.df_total[stock_ind]
        period_end = min((self.current_period + 1) * self.period_length, len(df))
        period_data = df.iloc[period_start:period_end]
        
        if period_data.empty:
            print("Warning: No data for this period. Skipping...")
            reward = 0
            done = True
        else:
            #print(self.strategies)
            #print(strategy)
            #print(period_data)
            bt = Backtest(period_data, self.strategies[strategy], cash=self.current_cash, commission=.001)
            results = bt.run()
            new_cash = results['Equity Final [$]']
            reward = (new_cash - self.current_cash) / self.current_cash
            #print(f"Action: {action}, Old Cash: {self.current_cash:.2f}, New Cash: {new_cash:.2f}, Reward: {reward:.6f}")
            
            self.current_cash = new_cash
                
        
        self.current_period += 1
        done = self.current_period >= self.total_periods
        
        return self._get_state(), reward, done

    def reset(self):
        self.current_period = 0
        self.current_cash = self.initial_cash
        return self._get_state()