import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class Discriminator(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.linear1 = Linear(2*num_nodes+1, 128)
        self.linear2 = Linear(128, 128)
        self.linear3 = Linear(128, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x
