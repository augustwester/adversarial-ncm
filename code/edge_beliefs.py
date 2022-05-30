import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

class EdgeBeliefs(nn.Module):
    """
    Class containing edge beliefs.
    """
    def __init__(self, num_nodes: int, temperature: float):
        super().__init__()
        self.num_nodes = num_nodes
        self.temperature = temperature
        self.P = Parameter(torch.zeros(num_nodes, num_nodes).fill_diagonal_(-50))

    def categorical_sample(self, p: Tensor) -> Tensor:
        """
        Performs categorical sampling.

        This implementation is significantly faster than the one provided by
        PyTorch, since it does not require the instantiation of a new class
        when probabilities change.

        Args:
            p: A vector of probabilities for each category. Must sum to 1.

        Returns:
            The index of the sampled category
        """
        return (p.cumsum(-1) >= torch.rand(p.shape[:-1])[..., None]).byte().argmax(-1)

    @property
    def edge_beliefs(self) -> Tensor:
        """
        Returns edge beliefs in transposed format (row i containing outgoing
        edges for node i).
        """
        return torch.sigmoid(self.P.T)
        
    def sample_dags(self, num_dags: int) -> Tensor:
        """
        Samples DAGs from the current edge beliefs.

        The implementation follows the two-phase DAG sampling proposed by
        Scherrer et al. (https://arxiv.org/abs/2109.02429)

        Args:
            num_dags: Number of dags to sample

        Returns:
            A tensor of size num_dags x N x N containing num_dags adjacency
            matrices, each representing a samples DAG.
        """
        P = torch.sigmoid(self.P.unsqueeze(0).repeat(num_dags, 1, 1))
        rem_idxs = torch.arange(self.num_nodes).unsqueeze(0).repeat(num_dags, 1)
        order = torch.empty(num_dags, self.num_nodes).long()
        
        for i in range(self.num_nodes):
            max_rows, _ = torch.max(P, dim=2)
            inv_max = (1 - max_rows)
            probs = F.softmax(inv_max / self.temperature, dim=1)
            idx = self.categorical_sample(probs)
            order[:, i] = rem_idxs[torch.arange(num_dags), idx]
            
            rem_mask = torch.ones_like(rem_idxs)
            rem_mask = rem_mask.scatter_(1, idx.view(-1, 1), 0).bool()
            rem_idxs = rem_idxs[rem_mask].view(num_dags, self.num_nodes-i-1)
            
            P_mask = torch.full(P.shape, True, dtype=torch.bool)
            zero_to_n = torch.arange(P.size(0))
            P_mask[zero_to_n, :, idx] = False
            P_mask[zero_to_n, idx, :] = False
            P = P[P_mask].view(num_dags, self.num_nodes-i-1, self.num_nodes-i-1)

        idx = order.view(num_dags, self.num_nodes, 1).repeat(1, 1, self.num_nodes)
        
        A = self.P.unsqueeze(0).repeat(num_dags, 1, 1)
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
