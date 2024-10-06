import pandas as pd
import numpy as np
import math

def portion(x):
    pos_steepness = 15
    neg_steepness = 10
    if x >= 0:
        return math.tanh(pos_steepness * x)
    else:
        return math.tanh(neg_steepness * x)

class TradingEnvironment:
    def __init__(self, df, initial_cash=10000, period_length=5, total_period=100):
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

    def _get_state(self):
        if self.current_period == 0:
            period_data = self.df.iloc[0:1]
        else:
            period_start = (self.current_period - 8) * self.period_length
            period_end = self.current_period * self.period_length
            period_data = self.df.iloc[period_start:period_end]
        
        if period_data.empty:
            return np.zeros(16)  # Increased to 16 to accommodate the new feature
    
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
            return self._get_state(), True, 0, self.final_equity / self.initial_equity - 1

        period_start = self.current_period * self.period_length
        period_end = min((self.current_period + 1) * self.period_length, len(self.df))
        
        period_data = self.df.iloc[period_start:period_end]
        if period_data.empty:
            print("Warning: No data for this period. Skipping...")
            return self._get_state(), True, 0, self.final_equity / self.initial_equity - 1

        trade_portion = portion(action)
        start_price = period_data['Open'].iloc[0]
        end_price = period_data['Close'].iloc[-1]

        # Determine optimal action
        if end_price > start_price:
            optimal_action = 1  # Buy
        elif end_price < start_price:
            optimal_action = -1  # Sell
        else:
            optimal_action = 0  # Hold

        current_equity = self.current_cash + self.shares * start_price

        if trade_portion >= 0:  # Buy
            spend = self.current_cash * trade_portion
            self.shares += spend // start_price
            self.current_cash -= (spend // start_price) * start_price
        else:  # Sell
            sell = int(self.shares * abs(trade_portion))
            self.current_cash += sell * start_price
            self.shares -= sell

        self.final_equity = self.current_cash + self.shares * end_price
        
        self.current_period += 1
        done = self.current_period >= self.total_periods
        
        print(f"period: {self.current_period}  action: {action:.4f}  trade_portion: {trade_portion:.4f}  cash: {self.current_cash:.2f}  shares: {self.shares}  close: {end_price:.2f}  equity: {self.final_equity:.2f}  optimal_action: {optimal_action}")

        return self._get_state(), done, optimal_action, self.final_equity / self.initial_equity - 1

    def reset(self):
        self.current_period = 8
        self.current_cash = self.initial_cash
        self.shares = 0
        self.initial_equity = self.initial_cash
        self.final_equity = self.initial_cash
        return self._get_state()