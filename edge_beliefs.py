import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class EdgeBeliefs(nn.Module):
    def __init__(self, num_nodes, temperature):
        super().__init__()
        self.num_nodes = num_nodes
        self.temperature = temperature
        self.p = Parameter(torch.zeros(num_nodes, num_nodes).fill_diagonal_(-10))

    def categorical_sample(self, p):
        return (p.cumsum(-1) >= torch.rand(p.shape[:-1])[..., None]).byte().argmax(-1)
        
    def sample_dags(self, num_dags):
        p = self.p.unsqueeze(0).repeat(num_dags, 1, 1)
        rem_idxs = torch.arange(self.num_nodes).unsqueeze(0).repeat(num_dags, 1)
        order = torch.empty(num_dags, self.num_nodes).long()
        
        for i in range(self.num_nodes):
            max_rows, _ = torch.max(p, dim=2)
            inv_max = (1 - max_rows)
            probs = F.softmax(inv_max / self.temperature, dim=1)
            idx = self.categorical_sample(probs)
            order[:, i] = rem_idxs[torch.arange(num_dags), idx]
            
            rem_mask = torch.ones_like(rem_idxs)
            rem_mask = rem_mask.scatter_(1, idx.view(-1, 1), 0).bool()
            rem_idxs = rem_idxs[rem_mask].view(num_dags, self.num_nodes-i-1)
            
            p_mask = torch.full(p.shape, True, dtype=torch.bool)
            zero_to_n = torch.arange(p.size(0))
            p_mask[zero_to_n, :, idx] = False
            p_mask[zero_to_n, idx, :] = False
            p = p[p_mask].view(num_dags, self.num_nodes-i-1, self.num_nodes-i-1)

        idx = order.view(num_dags, self.num_nodes, 1).repeat(1, 1, self.num_nodes)
        
        A = self.p.unsqueeze(0).repeat(num_dags, 1, 1)
        A = torch.sigmoid(A)
        A = torch.gather(A, 1, idx)
        A = torch.gather(A, 2, idx.permute(0,2,1))
        A = A.tril(diagonal=-1)
        A = A.unsqueeze(1)
        A = torch.cat((A, 1-A), axis=1)
        A = torch.log(A)
        A = F.gumbel_softmax(A, hard=True, dim=1)[:, 0, :, :]
        A = torch.scatter(A, 1, idx, A.clone())
        A = torch.scatter(A, 2, idx.permute(0,2,1), A.clone())
        
        return A.permute(0, 2, 1), order
