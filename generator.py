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
        
    def forward(self, x, do_idx):
        A, order = self.edge_beliefs.sample_dags(self.num_dags)
        y = self.conditionals(x, A, order, do_idx)
        y = y.mean(0)
        do = torch.zeros_like(x)
        if do_idx is not None:
            do[:, do_idx] = 1
        return torch.cat((y, do), axis=1), A
