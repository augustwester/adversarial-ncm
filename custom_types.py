import numpy as np
from enum import Enum

class GraphType(Enum):
    CHAIN = "chain"
    COLLIDER = "collider"
    TREE = "tree"
    BIDIAG = "bidiag"
    JUNGLE = "jungle"
    FULL = "full"
    RANDOM = "random"

class FnType(Enum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"

AdjMatrix = np.matrix
