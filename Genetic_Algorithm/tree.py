class TreeNode:
    def __init__(self,val=0,height=1,ismut=False,left=None,right=None):
        '''
        this stores the value of the TreeNode.
        For operator it will be 0(AND),1(OR),2(NOT).
        For filter values it can take values from 0-15
        '''
        self.val=val
        self.left=left # denotes the left child of the node
        self.right=right # denotes the right child of the node
        self.height=height # denotes the height of the tree till this node
        self.ismut=ismut # Stores whether it's current state is mutated or not

    def __repr__(self):
            return f"TreeNode({self.val})"
    
