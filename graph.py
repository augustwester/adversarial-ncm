import numpy as np
from custom_types import GraphType

def make_graph(graph_type: GraphType, num_nodes: int) -> np.ndarray:
    """
    Constructs an adjacency matrix representing a structured or random graph.

    Args:
       graph_type: The type of graph (e.g. chain or ER-1)
       num_nodes: The number of nodes in the graph

    Returns:
        A num_nodes x num_nodes adjacency matrix
    """
    A = np.zeros((num_nodes, num_nodes))
    if graph_type is GraphType.CHAIN:
        for i in range(num_nodes-1):
            A[i, i+1] = 1
    elif graph_type is GraphType.COLLIDER:
        A[:, -1] = 1
        A[-1,-1] = 0
    elif graph_type is GraphType.TREE:
        for i in range(num_nodes):
            left_child = 2*i+1
            A[i, left_child:left_child+2] = 1
    elif graph_type is GraphType.JUNGLE:
        for i in range(num_nodes):
            left_child = 2*i+1
            left_grandchild = 2*left_child + 1
            A[i, left_child:left_child+2] = 1
            A[i, left_grandchild:left_grandchild+4] = 1
    elif graph_type is GraphType.BIDIAG:
        for i in range(num_nodes-1):
            A[i, i+1:i+3] = 1
    elif graph_type is GraphType.FULL:
        return np.triu(np.ones((num_nodes, num_nodes)), 1)
    elif graph_type is GraphType.ER1 or graph_type is GraphType.ER2:
        max_edges = num_nodes * (num_nodes - 1) / 2
        expected_edges = num_nodes if GraphType is GraphType.ER1 else 2*num_nodes
        assert expected_edges <= max_edges, "Expected number of edges cannot exceeed maximum number of edges"
        p = expected_edges / max_edges
        edges = np.random.choice([0,1], size=(num_nodes, num_nodes), p=[1-p, p])
        return np.triu(edges, k=1)
    return A
