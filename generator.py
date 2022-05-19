import torch
import numpy as np
from torch import nn
from edge_beliefs import EdgeBeliefs
from ncm import NCM

class Generator(nn.Module):
    """
    Wrapper class containing both edge beliefs and NCM
    """
    def __init__(self, num_nodes: int, temperature: float):
        """
        Initializes a generator

        Args:
            num_nodes: The number of nodes in the SCM being modeled
            temperature: The temperature used when sampling DAGs from the edge beliefs

        Returns:
            An generator object containing an initialized NCM and matrix of edge beliefs
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_beliefs = EdgeBeliefs(num_nodes, temperature)
        self.ncm = NCM(num_nodes)

    def forward(self,
                Z: torch.Tensor,
                A: torch.Tensor,
                order: torch.Tensor,
                do=None) -> torch.Tensor:
        """
        Computes samples from the associated NCM.

        Args:
            Z: An m x N matrix of noise values sampled from a prior distribution
            A: An N x N adjacency matrix. Must represent a DAG.
            order: The topological order in which the values should be computed
            do: An integer in the range [-1, N-1] specifying the interventional distribution. If None, a random intervention is picked for each sample.

        Returns:
            An m x N matrix of samples from the NCM
        """
        N = A.shape[1]

        if do is None:
            dos = torch.tensor(np.random.choice(list(range(-1, N)), size=len(Z)))
        else:
            dos = torch.ones(len(Z)).long() * do

        y = self.ncm(Z, A, order, dos)
        onehot = torch.zeros(Z.shape[0], Z.shape[1]+1)
        onehot[torch.arange(len(dos)), dos] = 1
        return torch.cat((y, onehot), axis=1)
