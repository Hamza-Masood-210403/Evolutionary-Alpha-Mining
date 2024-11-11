from scipy.stats import pearsonr
import pickle
import numpy as np


'''
Similarity defined as the pearson correlation between the portfolio series of the two alphas
'''
def similarity(pnl_arr1,pnl_arr2):
  l = min(len(pnl_arr1),len(pnl_arr2))
  if(l<=1):
    return 0
  if(np.all(pnl_arr1==pnl_arr1[0]) and np.all(pnl_arr2==pnl_arr2[0])):
    return 0
  elif(np.all(pnl_arr1==pnl_arr1[0]) or np.all(pnl_arr2==pnl_arr2[0])):
    return 1
  return pearsonr(pnl_arr1[:l],pnl_arr2[:l])[0]

def save(dicti, filepath):
  with open(filepath, 'wb') as pickle_file:
    pickle.dump(dicti, pickle_file)
    

def load(filepath):
  with open(filepath, 'rb') as pickle_file:
    dicti = pickle.load(pickle_file)
  return dicti
  
  
  

