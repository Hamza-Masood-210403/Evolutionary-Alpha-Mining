import numpy as np
from collections import deque
import random
from tree import TreeNode
import bisect

def print_tree(root):
    def get_height(root):
        if root is None:
            return 0
        return 1 + max(get_height(root.left), get_height(root.right))

    def fill(res, node, i, l, r):
        if node is None:
            return
        mid = (l + r) // 2
        res[i][mid] = str(node.val)
        # + ',' + str(node.ismut)
        fill(res, node.left, i + 1, l, mid - 1)
        fill(res, node.right, i + 1, mid + 1, r)

    height = get_height(root)
    width = (1 << height) - 1  # Full binary tree width
    res = [[" " for _ in range(width)] for _ in range(height)]
    fill(res, root, 0, 0, width - 1)

    for line in res:
        print("".join(line))
        
def bfs(root,depth):
    '''
    The BFS function is to search the node at a particular depth and at a
    particular horizontal coordinate of the tree(which we select randomly).
    This is done in lieu of crossover and mutation which we define later
    (Basically at a particular depth which node number to choose.)
    '''
    if depth == 0:
        return root
    queue = deque([root])
    while (queue and depth):
        x = len(queue)
        while(x):
            node=queue.popleft()
            if(node.left):
                queue.append(node.left)
            if(node.right):
                queue.append(node.right)
            x -= 1
        depth -= 1
    queue_len = len(queue)
    if(queue_len == 0):
        return None
    ind=np.random.randint(0,queue_len)
    while(ind):
        queue.popleft()
        ind -= 1
    return queue[0]

'''
The create_tree functions are a set of over loaded functions defined to create a new tree
using given children trees and operator nodes.The first one is for binary operator.
The second one is for unary operator
'''
def unary_create_tree(tree,op_node):
    # With NOT operator
    op_node.val = 2
    op_node.left = tree
    op_node.height = tree.height+1
    return op_node

def binary_create_tree(tree1,tree2,op_node):
    # With AND-OR operator
    r = np.random.randint(0,2)
    op_node.val = r
    op_node.left = tree1
    op_node.right = tree2
    op_node.height = tree1.height+1
    return op_node

def add_depth_binary(base_pop,n):
    cnt = len(base_pop)
    tot = (int)(0.5*cnt*(cnt-1))
    chosen_indices = random.sample(range(1,tot),n-1)
    base_pop_new = []

    # prefSum
    s = 0
    ps = []
    c = cnt - 1
    while c > 0:
      s+=c
      ps.append(s)
      c-=1

    for i in chosen_indices:
        left_id = bisect.bisect_right(ps, i)
        rr = i - ps[max(left_id-1,0)]
        right_id = left_id + rr
        root = TreeNode()
        root = binary_create_tree(base_pop[left_id], base_pop[right_id], root)
        base_pop_new.append(root)

    return base_pop_new

def add_depth_unary(base_pop,n):
  chosen_indices = random.sample(range(0,len(base_pop)),n)
  base_pop_new = []
  for i in chosen_indices:
        root = TreeNode()
        root = unary_create_tree(base_pop[i], root)
        base_pop_new.append(root)
  return base_pop_new
     