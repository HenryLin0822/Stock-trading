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


class TradingEnvironment:
    def __init__(self, df, strategies, initial_cash=10000, period_length=20):
        self.df = df
        self.strategies = strategies
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.current_period = 0
        self.period_length = period_length
        self.total_periods = len(df) // period_length
        
    def _get_state(self):
        if self.current_period == 0:
            # For the first period, we don't have previous data, so we'll use the first day's data
            period_data = self.df.iloc[0:1]
        else:
            # Use the previous period's data
            period_start = (self.current_period - 2) * self.period_length #consider past 40 days
            period_end = self.current_period * self.period_length
            period_data = self.df.iloc[period_start:period_end]
        
        if period_data.empty:
            return np.zeros(11)
        
        
        features_df = pd.DataFrame({
            'Open': period_data['Open'],
            'High': period_data['High'],
            'Low': period_data['Low'],
            'Close': period_data['Close'],
            'Volume': period_data['Volume'],
            'Money': period_data['Money'],
            'RSI': period_data['RSI'],
            'MACD': period_data['MACD'],
            'Signal': period_data['Signal'],
            'K':period_data['K'],
            'D':period_data['D']
        })
        
        #print(features_df)
        features_df = features_df.fillna(0)
        #print('features_df in _get_state:')
        #print(features_df)
       
        return features_df


    def step(self, action):
        if self.current_period >= self.total_periods:
            return self._get_state(), 0, True  # Return done if we've reached the end

        strategy = self.strategies[action]
        period_start = self.current_period * self.period_length
        period_end = min((self.current_period + 1) * self.period_length, len(self.df))
        
        period_data = self.df.iloc[period_start:period_end]
        #print first date
        #print(period_data.index[0])
        #print last date
        #print(period_data.index[-1])
        if period_data.empty:
            print("Warning: No data for this period. Skipping...")
            reward = 0
            done = True
        else:
            try:
                bt = Backtest(period_data, strategy, cash=self.current_cash, commission=.001)
                results = bt.run()
                
                new_cash = results['Equity Final [$]']
                reward = (new_cash - self.current_cash) / self.current_cash
                
                #print(f"Action: {action}, Old Cash: {self.current_cash:.2f}, New Cash: {new_cash:.2f}, Reward: {reward:.6f}")
                
                self.current_cash = new_cash
                
            except Exception as e:
                print(f"Error running backtest: {e}")
                reward = 0
        
        self.current_period += 1
        done = self.current_period >= self.total_periods
        
        return self._get_state(), reward, done

    def reset(self):
        self.current_period = 2
        self.current_cash = self.initial_cash
        return self._get_state()