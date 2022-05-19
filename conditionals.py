import torch
from torch import nn
from torch.nn import Sequential, Linear, LeakyReLU, ModuleList

class Conditionals(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.mlps = []
        for _ in range(num_nodes):
            mlp = Sequential(Linear(num_nodes+1, 32), LeakyReLU(), Linear(32, 1))
            self.mlps.append(mlp)
        self.mlps = ModuleList(self.mlps)

    def forward(self, x, A, order, do_idxs):
        batch_size = x.size(0)
        outputs = torch.zeros_like(x)

        num_interventions = (do_idxs != -1).count_nonzero()
        ones = torch.ones(num_interventions)
        #u = torch.normal(mean=2*ones, std=ones)
        u = torch.normal(mean=0*ones, std=ones)
        outputs[do_idxs != -1, do_idxs[do_idxs != -1]] = u

        for i in range(self.num_nodes):
            nodes = order[:, i]
            not_intervened = do_idxs != nodes
            nodes = order[not_intervened, i]
            masks = A[not_intervened, :, nodes]
            ins = masks * outputs[not_intervened]
            ins = torch.cat((ins, x[not_intervened, nodes].unsqueeze(1)), axis=1)
            unique, vals = torch.unique(nodes, return_inverse=True)
            for val in unique:
                outs = self.mlps[val](ins[nodes == val])
                subset = outputs[not_intervened]
                subset[nodes==val, val] = outs.squeeze()
                outputs[not_intervened] = subset
            """
            outs = self.mlps[i](ins)
            outputs[not_intervened, nodes] = outs.squeeze()
            """

        return outputs

    """
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
                    outputs[dag_idx, :, node_idx] = torch.normal(mean=2*torch.ones(batch_size), std=torch.ones(batch_size))
                    continue
                mask = A[dag_idx, :, node_idx]
                ins = mask * outputs[dag_idx].clone()
                ins = torch.cat((ins, x[:, node_idx].unsqueeze(1)), axis=1)
                outs = self.mlps[node_idx](ins)
                outputs[dag_idx, :, node_idx] = outs.squeeze()
        return outputs
    """
