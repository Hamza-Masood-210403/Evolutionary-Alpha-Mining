from tree_ops import add_depth_binary,add_depth_unary
from backtest import Backtest
import random
import numpy as np

'''
Initializes the buy and sell signals for warmstart
'''
def warm_start_init(pop_size,k,base_pop):
    cnt = len(base_pop)
    tot = (int)(0.5*cnt*(cnt-1))

    num_unary = np.random.randint(0,len(base_pop))
    num_binary = min(pop_size*k-num_unary,tot)
    binary_signals = add_depth_binary(base_pop,num_binary)
    unary_signals = add_depth_unary(base_pop,num_unary)

    return binary_signals + unary_signals

'''
Performs warmstart and returns the first generation of alphas
'''
def warmstart(dataset,base_buy_signals,base_sell_signals,buys,sells,n):
    buys = warm_start_init(n,2,buys)
    sells = warm_start_init(n,2,sells)

    fitness_arr = []
    for i in range(len(buys)):
        backtestobj = Backtest(buys[i], sells[i], dataset, base_buy_signals, base_sell_signals)
        _, fit = backtestobj.fitness_function()
        fitness_arr.append((fit, i))

    #fitness_arr = sorted(fitness_arr, key=lambda x: x[0], reverse = True) # sorts according to first key
    #fitness_arr = fitness_arr[:n] # selects top n elements

    rnd_smp = random.sample(range(len(fitness_arr)), n)
    fitness_arr = [fitness_arr[i] for i in rnd_smp]

    buys_new=[buys[t[1]] for t in fitness_arr]
    sells_new=[sells[t[1]] for t in fitness_arr]
    
    return buys_new, sells_new