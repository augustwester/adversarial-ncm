import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ModuleList

class Conditionals(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.mlps = []
        for _ in range(num_nodes):
            mlp = Sequential(Linear(num_nodes+1, 32), nn.LeakyReLU(), Linear(32, 1))
            self.mlps.append(mlp)
        self.mlps = ModuleList(self.mlps)
    
    def forward(self, x, A, order, do_idx):
        if do_idx is not None:
            A[:, :, do_idx] = 0
        num_dags = A.size(0)
        batch_size = x.size(0)
        outputs = torch.zeros(num_dags, batch_size, self.num_nodes)
        for dag_idx in range(num_dags):
            order_ = order[dag_idx]
            for node_idx in order_:
                if node_idx == do_idx:
                    outputs[dag_idx, :, node_idx] = x[:, node_idx]
                    continue
                mask = A[dag_idx, :, node_idx]
                ins = mask * outputs[dag_idx].clone()
                ins = torch.cat((ins, x[:, node_idx].unsqueeze(1)), axis=1)
                outs = self.mlps[node_idx](ins)
                outputs[dag_idx, :, node_idx] = outs.squeeze()
        return outputs
