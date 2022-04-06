import numpy as np
from numpy.random import randn, uniform, choice
from custom_types import FunctionType, AdjMatrix

class SCM(object):
    def __init__(self, A: AdjMatrix, fun_type: FunctionType, additive=True):
        super().__init__()
        self.A = A
        self.fun_type = fun_type
        self.num_nodes = A.shape[0]
        self.fns = []

        if fun_type is FunctionType.LINEAR:
            self.init_coeffs()
        elif fun_type is FunctionType.NONLINEAR:
            self.init_nns()

    def init_coeffs(self):
        for i in range(self.num_nodes):
            num_inputs = np.count_nonzero(self.A[:, i])
            coeffs = uniform(low=0.5, high=2, size=num_inputs)
            coeffs *= choice([-1, 1], size=num_inputs)
            fn = lambda X: coeffs * X
            self.fns.append(fn)

    def init_nns(self):
        num_hidden = 20
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        for i in range(self.num_nodes):
            num_inputs = np.count_nonzero(self.A[:, i])
            W_1 = randn(num_hidden, num_inputs)
            b_1 = randn(num_hidden, 1)
            W_2 = randn(1, num_hidden)
            b_2 = randn(1, 1)
            #fn = lambda X: np.tanh(W_2 @ np.tanh(W_1 @ X.T + b_1) + b_2)
            fn = lambda X: sigmoid(W_2 @ sigmoid(W_1 @ X.T + b_1) + b_2)
            self.fns.append(fn)

    def sample(self, num_samples, do=None):
        A = self.A.copy()
        if do is not None:
            A[:, do] = 0
        X = np.zeros((num_samples, self.num_nodes))
        for i in range(self.num_nodes):
            ins = X[:, np.argwhere(A[:, i])].reshape(num_samples, -1)
            u = randn(num_samples, 1) if self.fun_type is FunctionType.LINEAR else 0.1*randn(num_samples, 1)
            outs = self.fns[i](ins) + u if ins.any() else randn(num_samples)
            #outs = self.fns[i](ins) if ins.any() else randn(num_samples)
            X[:, i] = outs.flatten()
        return X
