import pandas as pd
import matplotlib.pyplot as plt

predictions = pd.read_csv('predictions.csv', index_col='date', parse_dates=True)
print(predictions)
close = pd.read_csv('../data/close_all.csv', index_col='date', parse_dates=True)
close = close['2330']
df = pd.DataFrame({
    'close': close
})
df_backtest = df['2020-01-02':'2024-07-19']
df_backtest['predictions'] = predictions['pred_5']
print(df_backtest)
plt.figure(figsize=(15, 6))
plt.plot(df_backtest.index, df_backtest['close'], label='Actual Close Prices')
plt.plot(df_backtest.index, df_backtest['predictions'], label='Predicted Close Prices')
plt.legend()
plt.title('Predicted vs Actual Close Prices')
plt.savefig('predictions_vs_actual.png')
plt.show()
plt.close()


