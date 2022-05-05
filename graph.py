import numpy as np
from custom_types import GraphType, AdjMatrix

def make_graph(type: GraphType, num_nodes: int) -> AdjMatrix:
    A = np.zeros((num_nodes, num_nodes))
    if type is GraphType.CHAIN:
        for i in range(num_nodes-1):
            A[i, i+1] = 1
    elif type is GraphType.COLLIDER:
        A[:, -1] = 1
        A[-1,-1] = 0
    elif type is GraphType.TREE:
        for i in range(num_nodes):
            left_child = 2*i+1
            A[i, left_child:left_child+2] = 1
    elif type is GraphType.JUNGLE:
        for i in range(num_nodes):
            left_child = 2*i+1
            left_grandchild = 2*left_child + 1
            A[i, left_child:left_child+2] = 1
            A[i, left_grandchild:left_grandchild+4] = 1
    elif type is GraphType.BIDIAG:
        for i in range(num_nodes-1):
            A[i, i+1:i+3] = 1
    elif type is GraphType.FULL:
        return np.triu(np.ones((num_nodes, num_nodes)), 1)
    elif type is GraphType.RANDOM1 or type is GraphType.RANDOM2:
        max_edges = num_nodes * (num_nodes - 1) / 2
        expected_edges = num_nodes if GraphType is GraphType.RANDOM1 else 2*num_nodes
        p = expected_edges / max_edges
        edges = np.random.choice([0,1], size=(num_nodes, num_nodes), p=[1-p, p])
        return np.triu(edges, k=1)
    return A
