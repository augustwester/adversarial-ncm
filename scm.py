import numpy as np
from numpy.random import randn, uniform, choice
from custom_types import FnType, AdjMatrix

class SCM(object):
    def __init__(self, A: AdjMatrix, fun_type: FnType, additive=True):
        super().__init__()
        self.A = A
        self.fun_type = fun_type
        self.num_nodes = A.shape[0]
        self.fs = []
        self.us = []

        if fun_type is FnType.LINEAR:
            self.init_coeffs()
        elif fun_type is FnType.NONLINEAR:
            self.init_nns()
        self.init_noise()

    def init_coeffs(self):
        for i in range(self.num_nodes):
            num_inputs = np.count_nonzero(self.A[:, i])
            coeffs = uniform(low=0.5, high=2, size=num_inputs)
            coeffs *= choice([-1, 1], size=num_inputs)
            fn = lambda X: np.sum(coeffs * X, axis=1)
            self.fs.append(fn)

    def init_nns(self):
        num_hidden = 20
        for i in range(self.num_nodes):
            num_inputs = np.count_nonzero(self.A[:, i])
            W_1 = randn(num_hidden, num_inputs)
            b_1 = randn(num_hidden, 1)
            W_2 = randn(1, num_hidden)
            b_2 = randn(1, 1)
            f = lambda X: W_2 @ np.tanh(W_1 @ X.T + b_1) + b_2
            self.fs.append(f)

    def init_noise(self):
        for i in range(self.num_nodes):
            var = np.random.uniform(low=1, high=2)
            u = lambda n: np.random.normal(loc=0, scale=np.sqrt(var), size=n)
            self.us.append(u)

    def sample(self, num_samples, do=None):
        A = self.A.copy()
        if do is not None:
            A[:, do] = 0
        X = np.zeros((num_samples, self.num_nodes))
        for i in range(self.num_nodes):
            ins = X[:, np.argwhere(A[:, i])].reshape(num_samples, -1)
            u = self.us[i](num_samples) if do == i else randn(num_samples)
            is_root = not ins.any()
            if self.fun_type == FnType.LINEAR:
                outs = u if is_root else self.fs[i](ins) + u
            elif self.fun_type == FnType.NONLINEAR:
                if is_root:
                    outs = u
                else:
                    outs = self.fs[i](ins) + 0.1*u
            X[:, i] = outs.flatten()
        return X

    def make_dataset(self, samples_per_intervention: int):
        X = np.zeros((0, 2*self.num_nodes))
        for i in range(-1, self.num_nodes):
            do = i if i >= 0 else None
            X_do_i = self.sample(num_samples=samples_per_intervention, do=do)
            onehot = np.zeros_like(X_do_i)
            if do is not None: onehot[:, do] = 1
            X_do_i = np.concatenate((X_do_i, onehot), axis=1)
            X = np.concatenate((X, X_do_i))
        np.random.shuffle(X)
        return X
