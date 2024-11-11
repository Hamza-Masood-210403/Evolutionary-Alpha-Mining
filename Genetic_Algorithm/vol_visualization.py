import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import path_dataset

#File Path consisting of filters of different time frames
file_path = path_dataset()
df = pd.read_csv(file_path)
df.drop('Date', axis = 1)
#the eur/usd pair rates
dataset = df['Close'].values

# Step 1: Calculate daily returns
close_prices=np.array(dataset)
daily_returns = np.diff(close_prices) / close_prices[:-1]

window = 100

# Step 2: Calculate rolling window-day volatility (standard deviation of returns)
rolling_volatility = np.array([np.std(daily_returns[i:i+window])  for i in range(0,len(daily_returns),window)])
roll_mean=np.mean(rolling_volatility)
rolling_volatility=np.array([min(1,r//(roll_mean*(1-0.1))) for r in rolling_volatility])

# plt.plot(close_prices, label='Close Prices', color='black')
# Add markers for each window-day range based on 0/1 in rolling_volatility
for i, vol in enumerate(rolling_volatility):
    start_idx = i * window
    end_idx = start_idx + window if start_idx + window < len(close_prices) else len(close_prices)
    marker = '.'
    color = 'red' if vol == 1 else 'green'
    plt.scatter(range(start_idx, end_idx), close_prices[start_idx:end_idx], marker=marker, color=color, s=10, label='High Volatility' if i == 0 else "Low Volatility" if i==(len(rolling_volatility)-1) else "")

# Labels and title
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('Close Prices with 100-day Range Volatility Labels')
plt.legend()
plt.show()