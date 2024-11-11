import pandas as pd
import os
import numpy as np
from tree import TreeNode
from backtest import Backtest

def preprocess(df, x = 16):
    base_buy_trees = [TreeNode(i) for i in range(x)]
    base_sell_trees = [TreeNode(i) for i in range(x)]

    base_buy_signals = (df[df.columns[1:1+x]].values).T
    base_sell_signals = (df[df.columns[1+2*x:1+3*x]].values).T
    
    
    return base_buy_trees,base_sell_trees,base_buy_signals,base_sell_signals

def path_dataset():
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, '..', 'multiframe_alpha_fil2.csv')
    
    return os.path.normpath(relative_path)

if __name__ == "__main__":
    file_path = path_dataset() 
    df = pd.read_csv(file_path)
    df = df.drop('Date', axis = 1)
    dataset = df['Close'].values

    base_buy_trees,base_sell_trees,base_buy_signals,base_sell_signals = preprocess(df)

    fitness_arr = []
    final_cap = []
    for i in range(len(base_buy_trees)):
        backtestobj = Backtest(base_buy_trees[i], base_sell_trees[i], dataset, base_buy_signals, base_sell_signals)
        cap, fit = backtestobj.sharpe_ratio()
        fitness_arr.append(fit)
        final_cap.append(cap[-1])
        print(f"Filter {i+1}: sharpe: {fit}, final capital: {cap[-1]}")
    
    print("Fitness (sharpe) of base filters:")
    print("Mean:", np.mean(fitness_arr))
    print("Max:", np.max(fitness_arr))
    print("Min:", np.min(fitness_arr))
    print("Final capital:")
    print("Mean:", np.mean(final_cap))
    print("Max:", np.max(final_cap))
    print("Min:", np.min(final_cap))

    
    
    