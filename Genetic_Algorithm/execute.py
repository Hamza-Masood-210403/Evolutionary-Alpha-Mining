from warmstart import warmstart
from gen_ops import simulated_next_generation
import numpy as np
from backtest import Backtest
from misc import similarity

'''
Algorithm to find the best strategy at a specific depth by initializing with a warm start and building higher generations iteratively.
'''
def best_strategy_at_depth_d(dataset, base_buy_signals, base_sell_signals, prev_buy_trees, prev_sell_trees, n=50, iterations=50, stopping_it=5):
  
  epsla, threshold = 0.001, 0.75  # Tolerance for stopping and similarity threshold
  buy_pop, sell_pop = warmstart(dataset, base_buy_signals, base_sell_signals, prev_buy_trees, prev_sell_trees, n)

  fit_arr, fitness_prev_arr, fitness_prev_pnl_arr = [], [], []
  best_fit, prev_fit, cnt, best_fit_id = 0, 0, 1, -1
  buy_opt, sell_opt = buy_pop, sell_pop

  # Initial fitness evaluation
  for index, (buy_tree, sell_tree) in enumerate(zip(buy_pop, sell_pop)):
    obj = Backtest(buy_tree, sell_tree, dataset, base_buy_signals, base_sell_signals)
    capital, fit = obj.fitness_function()
    fitness_prev_pnl_arr.append(capital)
    fitness_prev_arr.append((fit, index))

  # Penalize similar chromosomes for diversity
  for fit_val, index in fitness_prev_arr:
    cnt = sum(1 for _, index1 in fitness_prev_arr if index1 != index and similarity(fitness_prev_pnl_arr[index], fitness_prev_pnl_arr[index1]) > threshold)
    fitness_prev_arr[index] = (fit_val / (1.0 + cnt), index)

  # Sort chromosomes by fitness
  fitness_prev_arr.sort(key=lambda x: x[0], reverse=True)
  prev_fit = np.mean([t[0] for t in fitness_prev_arr[:10]])

  # Evolution loop
  for it in range(1, iterations):
    print(f"generation {it+1} started!!")
    buy_pop, sell_pop, next_fit, fitness_prev_arr, fitness_prev_pnl_arr = simulated_next_generation(
      base_buy_signals, base_sell_signals, buy_pop, sell_pop, dataset, it, iterations, fitness_prev_arr)

    fit_arr.append(next_fit)
    if next_fit > best_fit:
      best_fit, buy_opt, sell_opt, best_fit_id = next_fit, buy_pop, sell_pop, it

    # Early stopping if fitness improvement falls below threshold
    if abs(next_fit - prev_fit) <= epsla:
      cnt += 1
      if cnt == stopping_it:
        break
    else:
      cnt = 1
    prev_fit = next_fit

  return buy_opt, sell_opt, best_fit, np.array(fit_arr), best_fit_id


'''
Integrates the optimization algorithm across multiple depths, building and optimizing strategies at each depth level.
'''
def integrator(dataset, base_buy_signals, base_sell_signals, base_buy_trees, base_sell_trees, depth):
  buy_opt, sell_opt = base_buy_trees, base_sell_trees
  dicti = {}  # Stores results by depth

  for d in range(2, depth + 1):
    buy_opt, sell_opt, best_fit, fit_arr, best_fit_id = best_strategy_at_depth_d(
      dataset, base_buy_signals, base_sell_signals, buy_opt, sell_opt)
    print("DEPTH AND BEST_FITNESS of POPULATION AND ITER: ", d, best_fit, best_fit_id)

  return dicti
