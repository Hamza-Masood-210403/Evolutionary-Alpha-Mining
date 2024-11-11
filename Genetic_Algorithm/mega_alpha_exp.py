from backtest import Backtest
from backtest_signal import Backtest as back_test
from misc import load
from main import path_dataset, preprocess
from signal_gen import tree_signal
import pandas as pd
import os

def path():
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'dic.pkl')
    
    return os.path.normpath(relative_path)

fp = path() # loads dic.pkl
dicti = load(fp)
max_depth = max(dicti.keys())

file_path = path_dataset()
df = pd.read_csv(file_path)
df = df.drop('Date', axis = 1)
dataset = df['Close'].values


'''
Generates mega alpha by taking the weighted average of the signals provided by the strategies in the pool
'''

_, _, base_buy_signals, base_sell_signals = preprocess(df)
window = 50 # use to judge the alpha fitness
for d in range(2,max_depth+1):
    print("Depth = ",d)
    buy_opt, sell_opt, _, _ = dicti[d]
    buy_opt, sell_opt, best_fit, fit_arr = dicti[d]
    sharpe_arr = []
    for i in range(len(buy_opt)):
        backtestobj = Backtest(buy_opt[i], sell_opt[i], dataset, base_buy_signals, base_sell_signals)
        _, sp=backtestobj.fitness_function()
        sharpe_arr.append([sp,i])

    # taking the top 20 alphas in the current pool
    sharpe_arr=sorted(sharpe_arr,key=lambda x: x[0], reverse = True)
    indx=[i[1] for i in sharpe_arr]
    indx=indx[:20]
    wt=[i[0] for i in sharpe_arr]
    wt = wt[:20]
    
    # Generating the mega signal for depth d
    mega_signal = [0]
    for day in range(1,len(dataset)):
        weighted_signal = 0
        j = 0
        sp_norm_const = 0
        for i in indx:
            signal = [buy - sell for buy, sell in zip(tree_signal(base_buy_signals[:,day-1:day+1], buy_opt[i]), tree_signal(base_sell_signals[:,day-1:day+1], sell_opt[i]))]
            # adjusting the weights after some days
            if(day % window == 0):
                backtestobj = Backtest(buy_opt[i], sell_opt[i], dataset[day-window:day], base_buy_signals[:,day-window:day], base_sell_signals[:,day-window:day])
                cap,sp=backtestobj.fitness_function()
                wt[j] += sp
            weighted_signal += wt[j]*signal[-1]
            sp_norm_const += wt[j]
            j += 1
    
        if sp_norm_const == 0:
            sp_norm_const = 1
        mega_signal.append(weighted_signal/sp_norm_const)

    print("Training results:")
    backtestobj = back_test(dataset[:4000], mega_signal[:4000])
    cap,sp=backtestobj.sharpe_ratio()
    print(cap[-1],sp)
    print("Testing results:")
    backtestobj = back_test(dataset[4000:], mega_signal[4000:])
    cap,sp=backtestobj.sharpe_ratio()
    print(cap[-1],sp)