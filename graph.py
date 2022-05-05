import numpy as np
from custom_types import GraphType, AdjMatrix

def make_structured_graph(type: GraphType, num_nodes: int) -> AdjMatrix:
    A = np.zeros((num_nodes, num_nodes))
    if type is GraphType.CHAIN:
        for i in range(num_nodes-1):
            A[i, i+1] = 1
    elif type is GraphType.COLLIDER:
        A[:, -1] = 1
        A[-1,-1] = 0
    elif type is GraphType.TREE or type is GraphType.JUNGLE:
        # TO-DO: Jungle is Tree right now
        num_levels = 1 + int(np.log2(num_nodes))
        for i in range(num_levels-1): # skip the last level
            num_nodes_before_level = 2**i - 1
            num_nodes_at_level = min(2**i, num_nodes-num_nodes_before_level)
            for j in range(num_nodes_at_level):
                first_child = num_nodes_before_level + num_nodes_at_level + 2*j
                A[num_nodes_before_level+j, first_child:first_child+2] = 1
    elif type is GraphType.BIDIAG:
        for i in range(num_nodes-1):
            A[i, i+1:i+3] = 1
    return A

def make_random_graph(num_nodes, e):
    max_edges = num_nodes * (num_nodes - 1) / 2
    expected_edges = num_nodes * e
    p = expected_edges / max_edges
    edges = np.random.choice([0,1], size=(num_nodes, num_nodes), p=[1-p, p])
    return np.triu(edges, k=1)
