import numpy as np
import random
import copy
from tree_ops import bfs
from misc import similarity
from backtest import Backtest

'''
If a tree is to undergo mutation we select a random node where we need to do
the mutation using bfs and set it's mutate to not of the previous value.
'''
def mutation(root):
    l = root.height
    r = np.random.randint(0,l)
    node = bfs(root,r)
    if(node is not None):
        if(node.height == 1):
            node.val = np.random.randint(0,16)
        else:
            if(node.val == 1):
                node.val = 0
            else:
                node.val = 1
        node.ismut = not node.ismut

'''
  Swaps left or right subtrees for nodes with NOT operators.
'''
def not_rootswap(root_a,root_b):
    x = root_a.left
    r = np.random.randint(0,2)
    if r:
        root_a.left = root_b.right
        root_b.right = x
    else:
        root_a.left = root_b.left
        root_b.left = x

'''
  Swaps the entire root nodes.
'''
def gen_rootswap(root_a,root_b):
    x = root_a
    root_a = root_b
    root_b = x

#This is the crossover between 2 buy or 2 sell signals at any random node at a particular depth d.
def crossover(tree1,tree2):
    h = tree1.height
    depth = np.random.randint(0,h-1) # range = [0 to h-2]
    child1 = copy.deepcopy(tree1)
    child2 = copy.deepcopy(tree2)
    root1 = bfs(child1,depth)
    root2 = bfs(child2,depth)

    if root1 is not None and root2 is not None:
      # if both NOT operator
      if root1.val==2 and root2.val==2:
        x = root1.left
        root1.left = root2.left
        root2.left = x
      # if root1 is NOT operator
      elif root1.val == 2:
        not_rootswap(root1,root2)
      # if root2 is NOT operator
      elif root2.val == 2:
        not_rootswap(root2,root1)
      # if none of them are NOT operator
      else:
        swap_choice = random.choice(['left_left', 'right_right', 'left_right', 'right_left'])
        if swap_choice == 'left_left':
          gen_rootswap(root1.left,root2.left)
        if swap_choice == 'left_right':
          gen_rootswap(root1.left,root2.right)
        if swap_choice == 'right_left':
          gen_rootswap(root1.right,root2.left)
        if swap_choice == 'right_right':
          gen_rootswap(root1.right,root2.right)

    return child1,child2

# tournament selection to determine the parents
def tournament_selection(fitness_arr, k = 3):
    tournament = random.sample(list(enumerate(fitness_arr)), k)
    winner = max(tournament, key=lambda x: x[1][0])
    id = winner[1][1]
    return id

'''
Generates the next generation through crossover, mutation, and selection, penalizing similar chromosomes.
'''
def simulated_next_generation(base_buy, base_sell, buy_tree_pop, sell_tree_pop, dataset, curr_gen, tot_gen, fitness_arr):
  threshold=0.75
  mul = np.exp(-curr_gen/tot_gen)
  sz = len(fitness_arr)
  fit = np.array([t[0] for t in fitness_arr])

  buy_tree_pop_new,sell_tree_pop_new=[],[]

  # Preserve top performers (elitism)
  n = len(fitness_arr)//100
  for i in range(n):
    id = fitness_arr[i][1]
    buy_tree_pop_new.append(buy_tree_pop[id])
    sell_tree_pop_new.append(sell_tree_pop[id])

  # Create new individuals through crossover and mutation
  for i in range((sz-n)//2):
    id1 = tournament_selection(fitness_arr)
    id2 = tournament_selection(fitness_arr)
    bpar1 = buy_tree_pop[id1]
    spar1 = sell_tree_pop[id1]
    bpar2 = buy_tree_pop[id2]
    spar2 = sell_tree_pop[id2]

    if random.random() < 0.6:
      # Evaluate offspring and select the best ones
      child1,child2=crossover(bpar1,bpar2)
      child11,child22=crossover(spar1,spar2)
      obj1 = Backtest(child1,child11,dataset,base_buy,base_sell)
      obj2 = Backtest(child2,child22,dataset,base_buy,base_sell)
      obj3 = Backtest(child2,child11,dataset,base_buy,base_sell)
      obj4 = Backtest(child1,child22,dataset,base_buy,base_sell)
      objp1 = Backtest(bpar1,spar1,dataset,base_buy,base_sell)
      objp2 = Backtest(bpar2,spar2,dataset,base_buy,base_sell)
      _,fit1 = obj1.fitness_function()
      _,fit2 = obj2.fitness_function()
      _,fit3 = obj3.fitness_function()
      _,fit4 = obj4.fitness_function()
      _,fitp1 = objp1.fitness_function()
      _,fitp2 = objp2.fitness_function()

      # Select the most fit pair of children or keep parents if offspring fitness is lower
      if(max(fit1,fit2)>min(fitp1,fitp2)):
        buy_tree_pop_new.append(child1)
        sell_tree_pop_new.append(child11)
        buy_tree_pop_new.append(child2)
        sell_tree_pop_new.append(child22)
      elif(max(fit3,fit4)>min(fitp1,fitp2)):
        buy_tree_pop_new.append(child1)
        sell_tree_pop_new.append(child22)
        buy_tree_pop_new.append(child2)
        sell_tree_pop_new.append(child11)
      else:
        buy_tree_pop_new.append(bpar1)
        sell_tree_pop_new.append(spar1)
        buy_tree_pop_new.append(bpar2)
        sell_tree_pop_new.append(spar2)
    else:
        buy_tree_pop_new.append(bpar1)
        sell_tree_pop_new.append(spar1)
        buy_tree_pop_new.append(bpar2)
        sell_tree_pop_new.append(spar2)

  # Mutate selected individuals
  n_mut_arr=np.random.randint(0,len(buy_tree_pop_new),size=len(buy_tree_pop_new)//10)
  for i in n_mut_arr:
    mutation(buy_tree_pop_new[i])
  n_mut_arr1=np.random.randint(0,len(buy_tree_pop_new),size=len(buy_tree_pop_new)//10)
  for i in n_mut_arr1:
    mutation(sell_tree_pop_new[i])

  # Evaluate new population fitness and penalize similarity
  fitness_new_arr,fitness_new_pnl_arr=[],[]
  for index, (buy_tree, sell_tree) in enumerate(zip(buy_tree_pop_new, sell_tree_pop_new)):
    obj=Backtest(buy_tree,sell_tree,dataset,base_buy,base_sell)
    capital,fit=obj.fitness_function()
    fitness_new_pnl_arr.append(capital)
    fitness_new_arr.append((fit,index))
  fitness_arr_gen_new=fitness_new_arr

  # Apply shared fitness adjustment for similar chromosomes
  for fit_val,index in fitness_new_arr:
    fit_new=fit_val
    cnt=0
    for fit1,index1 in fitness_new_arr:
      if(index1!=index and similarity(fitness_new_pnl_arr[index],fitness_new_pnl_arr[index1])>threshold):
        cnt+=1
    if((1.0+cnt)*mul) >= 1:
       fit_new=fit_new/((1.0+cnt)*mul)
    fitness_arr_gen_new[index]=fit_new,index
  fitness_new_arr=fitness_arr_gen_new

  # Sort population by fitness and calculate average fitness for the top individuals
  fitness_new_arr = sorted(fitness_new_arr, key=lambda x: x[0], reverse = True)

  fit_gen_new = np.array([t[0] for t in fitness_new_arr])
  print(len(fitness_new_arr),np.mean(fit_gen_new))
  avg_new_fitness_pop=np.mean(fit_gen_new[:10])
  return buy_tree_pop_new,sell_tree_pop_new,avg_new_fitness_pop,fitness_new_arr,fitness_new_pnl_arr