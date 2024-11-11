import numpy as np

'''
Generates the signal of any given tree by evaluating the base filters through the applied operators
'''
def tree_signal(base_signals,node):
  if(node == None):
    return np.zeros(len(base_signals[0]))

  if(node.left == None and node.right == None):
    return base_signals[node.val]

  signal = np.zeros(len(base_signals[0]))

  if(node.val == 2):
    child_signal = tree_signal(base_signals,node.left)
    signal = np.logical_not(child_signal).astype(int)

  else:
    left_child_signal = tree_signal(base_signals,node.left)
    right_child_signal = tree_signal(base_signals,node.right)
    if(node.val == 1):
      signal = np.logical_or(left_child_signal,right_child_signal).astype(int)
    else:
      signal = np.logical_and(left_child_signal,right_child_signal).astype(int)

  return signal