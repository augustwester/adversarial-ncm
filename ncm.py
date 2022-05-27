import torch
from torch import nn, Tensor
from torch.nn import Sequential, Linear, LeakyReLU, ModuleList

class NCM(nn.Module):
    """
    Class implementing an NCM.

    See Xia et al. (2021): https://arxiv.org/abs/2107.00793
    """
    def __init__(self, num_nodes: int):
        """
        Initializes the NCM.

        Args:
            num_nodes: The number of nodes in the NCM

        Returns:
            An initialized NCM
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.mlps = []
        for _ in range(num_nodes):
            mlp = Sequential(Linear(num_nodes+1, 32), LeakyReLU(), Linear(32, 1))
            self.mlps.append(mlp)
        self.mlps = ModuleList(self.mlps)

    def forward(self, Z: Tensor, A: Tensor, order: Tensor, do: Tensor) -> Tensor:
        """
        Computes samples from the NCM given a graph and a matrix of noise vectors.

        Args:
            Z: An m x N matrix of noise values sampled from a prior distribution
            A: An N x N adjacency matrix. Must represent a DAG.
            order: The topological order in which the values should be computed
            do: An m-dimensional vector with integer entries in the range [-1, N-1] specifying the interventional distribution for each sample

        Returns:
            An m x N matrix of samples from the NCM
        """
        outputs = torch.zeros_like(Z)
        num_interventions = (do != -1).count_nonzero()
        ones = torch.ones(num_interventions)
        u = torch.normal(mean=2*ones, std=ones)
        outputs[do != -1, do[do != -1]] = u

        for i in range(self.num_nodes):
            nodes = order[:, i]
            not_intervened = do != nodes
            nodes = order[not_intervened, i]
            masks = A[not_intervened, :, nodes]
            ins = masks * outputs[not_intervened]
            ins = torch.cat((ins, Z[not_intervened, nodes].unsqueeze(1)), axis=1)
            unique = torch.unique(nodes)
            for val in unique:
                outs = self.mlps[val](ins[nodes == val])
                subset = outputs[not_intervened]
                subset[nodes==val, val] = outs.squeeze()
                outputs[not_intervened] = subset

        return outputs
