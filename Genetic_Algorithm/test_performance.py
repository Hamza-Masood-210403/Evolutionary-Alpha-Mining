from backtest import Backtest
from misc import load
from main import path_dataset,preprocess
from tree_ops import print_tree
import os
import pandas as pd
import numpy as np

def path():
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'dic.pkl')
    
    return os.path.normpath(relative_path)

fp = path() # loads dic.pkl
dicti = load(fp)
max_depth = max(dicti.keys())

file_path = path_dataset() # loads filters_data.csv
df = pd.read_csv(file_path)
df = df.drop('Date', axis = 1)
df = df[4000:]
dataset = df['Close'].values

'''
Tests the performace of the generated alphas by backtesting them on the test dataset
'''

_, _, base_buy_signals,base_sell_signals = preprocess(df)
plot_cap=[]
for d in range(2,max_depth+1):
    buy_opt, sell_opt, best_fit, fit_arr = dicti[d]
    print(f"Depth: {d}")
    sharpe_arr = []
    max_cap = -1
    for i in range(len(buy_opt)):
        backtestobj = Backtest(buy_opt[i], sell_opt[i], dataset, base_buy_signals, base_sell_signals)
        cap, sp=backtestobj.sharpe_ratio()
        sharpe_arr.append(sp)
        if(max_cap == -1 or max_cap<cap[-1]):
            max_cap = cap[-1]

    idx = np.argmax(sharpe_arr)
    sharpe_arr=sorted(sharpe_arr,reverse=True)
    print('avg top 10 alphas',np.mean(np.array(sharpe_arr[:10])))
    print(f'Avg Sharpe = {np.mean(np.array(sharpe_arr))}')
    print(f'Max Sharpe = {np.max(np.array(sharpe_arr))}')
    print(f'Min Sharpe = {np.min(np.array(sharpe_arr))}')
    print(f'Maximum value of the portfolio = {max_cap}')

