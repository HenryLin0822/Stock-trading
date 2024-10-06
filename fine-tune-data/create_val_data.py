import pandas as pd
import backtesting
from strategies import *
from datasets import Dataset
close = pd.read_csv('../data/close_all.csv', index_col='date', parse_dates=True)
open_p = pd.read_csv('../data/open_all.csv', index_col='date', parse_dates=True)
high = pd.read_csv('../data/high_all.csv', index_col='date', parse_dates=True)
low = pd.read_csv('../data/low_all.csv', index_col='date', parse_dates=True)
volume = pd.read_csv('../data/num_all.csv', index_col='date', parse_dates=True)
money = pd.read_csv('../data/money_all.csv', index_col='date', parse_dates=True)
print('Data loaded')
stock = '2330'
def df_to_readable_string(df):
    # Reset index to make date a column
    df_reset = df.reset_index()
    
    # Convert date to string and format other columns
    df_reset['date'] = df_reset['date'].dt.strftime('%Y-%m-%d')
    df_reset['Money'] = df_reset['Money'].apply(lambda x: f"{x:.2e}")
    
    # Create a list of dictionaries, each representing a row
    rows = df_reset.to_dict('records')
    
    # Format each row as a string
    formatted_rows = [
        f"Date: {row['date']}, " +
        f"Close: {row['Close']:.1f}, " +
        f"Open: {row['Open']:.1f}, " +
        f"High: {row['High']:.1f}, " +
        f"Low: {row['Low']:.1f}, " +
        f"Volume: {row['Volume']:.0f}, " +
        f"Money: {row['Money']}"
        for row in rows
    ]
    
    # Join all rows into a single string
    return "\n".join(formatted_rows)
df = pd.DataFrame({
    'Close': close[stock],
    'Open': open_p[stock],
    'High': high[stock],
    'Low': low[stock],
    'Volume': volume[stock],
    'Money': money[stock]
})

df_train = df['2008-01-01':'2020-01-01']
df_val = df['2020-01-01':]
period_len = 20
val_total_periods = len(df_val)/period_len
strategies = [RSI, SmaCross, MACD, KD, BuyAndHold, MeanReversion]
instruction = '''Instructions: Analyze the given data for stock 2330 over the past four weeks (20 business days) and determine the best strategy to use for the next 4 weeks. Follow these steps:

1. Review the stock data:
   - Close: Closing price
   - Open: Opening price
   - High: Highest price of the day
   - Low: Lowest price of the day
   - Volume: Number of stocks traded
   - Money: Cash traded

2. Consider the following strategies:

   0. RSI Strategy:
      - Uses 14-day Relative Strength Index (RSI)
      - Buy when RSI crosses above 30 (oversold)
      - Sell when RSI crosses above 70 (overbought)
      - Aims to buy oversold and sell overbought assets

   1. SMA Cross Strategy:
      - Uses 10-day and 20-day Simple Moving Averages (SMA)
      - Buy when 10-day SMA crosses above 20-day SMA
      - Sell when 10-day SMA crosses below 20-day SMA
      - Identifies trend changes using short-term and medium-term averages

   2. KD Strategy (Stochastic Oscillator):
      - Uses 14-day window and 3-day smoothing
      - Buy when %K crosses above %D and %K is below 20 (oversold)
      - Sell when %K crosses below %D and %K is above 80 (overbought)
      - Identifies potential reversals in oversold and overbought conditions

   3. MACD Strategy:
      - Uses 12-day fast EMA, 26-day slow EMA, and 9-day signal line
      - Buy when MACD line crosses above signal line
      - Sell when MACD line crosses below signal line
      - Identifies trend changes and momentum shifts

   4. Buy and Hold Strategy:
      - Buy the entire portfolio if no current position
      - No explicit sell signal
      - Aims to capture long-term market growth
      - Avoids pitfalls of frequent trading (timing errors, transaction costs)

3. Analyze the data and choose the most appropriate strategy.

Output: Provide only a single number (0-4) representing the index of the chosen strategy.
Important: Your entire response must be only a single digit from 0 to 4, without any other characters, words, or punctuation. Do not include 'Output:' or any other text.
'''
val_data = []
current_period = 1
while(current_period < val_total_periods):
    df_period = df_val.iloc[current_period*period_len:(current_period+1)*period_len]
    df_previous_period = df_val.iloc[(current_period-1)*period_len:current_period*period_len]
    #print(df_previous_period.to_string())
    input_data = df_to_readable_string(df_previous_period)
    #print(temp['inputs'])

    val_data.append({
        "instruction": instruction,
        "input": input_data,
        "output":[]
    })
    #print(train_data[current_period-1])
    current_period += 1

dataset = Dataset.from_list(val_data)
dataset.save_to_disk("./"+str(stock)+"_val")
