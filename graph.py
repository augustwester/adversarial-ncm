import numpy as np
from custom_types import GraphType, AdjMatrix

def structured_adj_matrix(graph_type: GraphType, num_nodes: int) -> AdjMatrix:
    A = np.zeros((num_nodes, num_nodes))
    if graph_type is GraphType.CHAIN:
        for i in range(num_nodes-1):
            A[i, i+1] = 1
    elif graph_type is GraphType.COLLIDER:
        coll = np.random.randint(low=0, high=num_nodes)
        A[:, coll] = 1
        A[coll, coll] = 0
    elif graph_type is GraphType.TREE or graph_type is GraphType.JUNGLE:
        # TO-DO: Jungle is Tree right now
        num_levels = 1 + int(np.log2(num_nodes))
        for i in range(num_levels-1): # skip the last level
            num_nodes_before_level = 2**i - 1
            num_nodes_at_level = min(2**i, num_nodes-num_nodes_before_level)
            for j in range(num_nodes_at_level):
                first_child = num_nodes_before_level + num_nodes_at_level + 2*j
                A[num_nodes_before_level+j, first_child:first_child+2] = 1
    elif graph_type is GraphType.BIDIAG:
        for i in range(num_nodes-1):
            A[i, i+1:i+3] = 1
    return A
