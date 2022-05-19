import numpy as np
import torch
import torch.nn as nn
from edge_beliefs import EdgeBeliefs
from conditionals import Conditionals

class Generator(nn.Module):
    def __init__(self, num_nodes, num_dags, temperature=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_dags = num_dags
        self.edge_beliefs = EdgeBeliefs(num_nodes, temperature)
        self.conditionals = Conditionals(num_nodes)
        
    def forward(self, z, A, order, do_idx=None):
        N = A.shape[1]

        if do_idx is None:
            do_idxs = torch.tensor(np.random.choice(list(range(-1, N)), size=len(z)))
        else:
            do_idxs = torch.ones(len(z)).long() * do_idx

        y = self.conditionals(z, A, order, do_idxs)
        #do = torch.zeros_like(z)
        do = torch.zeros(z.shape[0], z.shape[1]+1)
        #do[do_idxs >= 0, do_idxs[do_idxs >= 0]] = 1
        do[torch.arange(len(do_idxs)), do_idxs] = 1
        return torch.cat((y, do), axis=1)

    """
    def forward(self, x, do_idx, A=None, order=None):
        if A is None and order is None:
            A, order = self.edge_beliefs.sample_dags(x.size(0))
        y = self.conditionals(x, A, order, do_idx)
        y = y.mean(0)
        do = torch.zeros_like(x)
        if do_idx is not None:
            do[:, do_idx] = 1
        return torch.cat((y, do), axis=1), A
    """
