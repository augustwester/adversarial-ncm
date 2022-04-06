import numpy as np
from enum import Enum

class GraphType(Enum):
    CHAIN = 1
    COLLIDER = 2
    TREE = 3
    BIDIAG = 4
    JUNGLE = 5
    FULL = 6
    RANDOM = 7

class FunctionType(Enum):
    LINEAR = 1
    NONLINEAR = 2

AdjMatrix = np.matrix
