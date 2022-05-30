import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Linear

class Discriminator(nn.Module):
    """
    Class implementing the discriminator
    """
    def __init__(self, num_nodes: int):
        """
        Initializes a discriminator with two 128-unit hidden layers.

        Args:
            num_nodes: The number of nodes in the SCM being modeled

        Returns:
            An initialized discriminator
        """
        super().__init__()
        self.linear1 = Linear(2*num_nodes+1, 128)
        self.linear2 = Linear(128, 128)
        self.linear3 = Linear(128, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the output of the discriminator.

        Args:
            x: An m x 2N+1 matrix of inputs.

        Returns:
            A value in the range (0,1) for each of the m samples. The value
            represents the discriminator's belief that the sample is from the
            true SCM and not the NCM.
        """
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x
