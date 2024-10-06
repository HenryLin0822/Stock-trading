import pandas as pd
import numpy as np
import math

class TradingEnvironment:
    def __init__(self, df, initial_cash=10000, period_length=5, total_period=100, n=10):
        self.df = df
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.current_period = 0
        self.period_length = period_length
        self.total_period = total_period
        self.total_periods = len(df) // period_length
        self.shares = 0
        self.initial_equity = initial_cash
        self.final_equity = initial_cash
        self.n = n
        self.action_space = list(range(-n, n+1))

    def _get_state(self):
        if self.current_period == 0:
            period_data = self.df.iloc[0:1]
        else:
            period_start = (self.current_period - 8) * self.period_length
            period_end = self.current_period * self.period_length
            period_data = self.df.iloc[period_start:period_end]
        
        if period_data.empty:
            return np.zeros(16)
    
        features_df = pd.DataFrame({
            'Open': period_data['Open'],
            'Close': period_data['Close'],
            'Predictions5': period_data['Predictions5'],
            'Predictions10': period_data['Predictions10'],
            'TimePortion': [(self.current_period * self.period_length) / self.total_period] * len(period_data)
        })
        
        features_df = features_df.fillna(0)
       
        return features_df

    def step(self, action):
        if self.current_period >= self.total_periods:
            return self._get_state(), 0, True

        period_start = self.current_period * self.period_length
        period_end = min((self.current_period + 1) * self.period_length, len(self.df))
        
        period_data = self.df.iloc[period_start:period_end]
        if period_data.empty:
            print("Warning: No data for this period. Skipping...")
            return self._get_state(), 0, True

        start_price = period_data['Open'].iloc[0]
        end_price = period_data['Close'].iloc[-1]

        current_equity = self.current_cash + self.shares * start_price

        action_value = self.action_space[action]
        if action_value > 0:  # Buy
            max_shares = self.current_cash // start_price
            shares_to_buy = min(max_shares, int(max_shares * (action_value / self.n)))
            cost = shares_to_buy * start_price
            self.shares += shares_to_buy
            self.current_cash -= cost
        elif action_value < 0:  # Sell
            shares_to_sell = min(self.shares, int(self.shares * (abs(action_value) / self.n)))
            revenue = shares_to_sell * start_price
            self.shares -= shares_to_sell
            self.current_cash += revenue

        self.final_equity = self.current_cash + self.shares * end_price
        
        self.current_period += 1
        done = self.current_period >= self.total_periods
        
        reward = (self.final_equity - current_equity) / current_equity

        return self._get_state(), reward, done

    def reset(self):
        self.current_period = 8
        self.current_cash = self.initial_cash
        self.shares = 0
        self.initial_equity = self.initial_cash
        self.final_equity = self.initial_cash
        return self._get_state()