import pandas as pd
import os
import numpy as np
import random
from tree import TreeNode
from execute import integrator
from misc import save
from tree_ops import print_tree


'''
Function to construct the base nodes of the tree and load base buy and sell signals
'''
def preprocess(df, x = 32):
    base_buy_trees = [TreeNode(i) for i in range(x)]
    base_sell_trees = [TreeNode(i) for i in range(x)]

    base_buy_signals = (df[df.columns[1:1+x]].values).T
    base_sell_signals = (df[df.columns[1+x:1+2*x]].values).T
    
    
    return base_buy_trees,base_sell_trees,base_buy_signals,base_sell_signals

def path_dataset():
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, '..', 'multiframe_alpha_fil2.csv')
    
    return os.path.normpath(relative_path)

def run(dicti,depth):
    for d in range(2,depth):
        buy_opt,sell_opt,best_fit,fit_arr = dicti[d]
        print(f"Avg fitness of top 10 individuals of population at depth {d}:", best_fit)

if __name__ == "__main__":
    file_path = path_dataset() 
    df = pd.read_csv(file_path)
    df = df.drop('Date', axis = 1)
    df = df[:4000]
    dataset = df['Close'].values
    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)

    dataset = df['Close'].values

    base_buy_trees,base_sell_trees,base_buy_signals,base_sell_signals = preprocess(df)

    depth = 6
    
    # Runs the algorithm and saves the results in a dictionary
    dicti = integrator(dataset, base_buy_signals, base_sell_signals, base_buy_trees, base_sell_trees, depth)

    save(dicti, 'dic.pkl')
    run(dicti,7)
    