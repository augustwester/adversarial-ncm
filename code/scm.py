import numpy as np
from numpy.random import randn, uniform, choice, normal
from custom_types import FnType

class SCM():
    """
    Class for constructing random linear or nonlinear SCMs obeying a
    predefined causal graph.
    """
    def __init__(self, A: np.ndarray, fn_type: FnType):
        """
        Initializes the SCM.

        Args:
            A: Adjacency matrix of the causal graph. Must be a DAG.
            fn_type: Specifies whether the functional relationships should be linear or nonlinear

        Returns:
            An initialized SCM
        """
        super().__init__()
        self.A = A
        self.fn_type = fn_type
        self.num_nodes = A.shape[0]
        self.fs = []
        self.us = []

        if fn_type is FnType.LINEAR:
            self.init_coeffs()
        elif fn_type is FnType.NONLINEAR:
            self.init_nns()
        self.init_noise()

    def init_coeffs(self):
        """
        Initializes random coefficients in the intervals [-1, -0.25] U [0.25, 1].
        Only used for linear SCMs.
        """
        for i in range(self.num_nodes):
            num_inputs = np.count_nonzero(self.A[:, i])
            coeffs = uniform(low=0.25, high=1, size=num_inputs)
            coeffs *= choice([-1, 1], size=num_inputs)
            fn = lambda X, coeffs=coeffs: np.sum(coeffs * X, axis=1)
            self.fs.append(fn)

    def init_nns(self):
        """
        Initializes random neural networks with 20 hidden units to represent
        the functional relationships of nonlinear SCMs.
        """
        num_hidden = 10
        leaky_relu = lambda x: np.where(x > 0, x, 0.25*x)
        for i in range(self.num_nodes):
            num_inputs = np.count_nonzero(self.A[:, i])
            W_1 = randn(num_hidden, num_inputs)
            b_1 = randn(num_hidden, 1)
            W_2 = randn(1, num_hidden)
            b_2 = randn(1, 1)
            f = lambda X, W_1=W_1, b_1=b_1, W_2=W_2, b_2=b_2: W_2 @ leaky_relu(W_1 @ X.T + b_1) + b_2
            self.fs.append(f)

    def init_noise(self):
        """
        Initializes the distributions used for sampling additive or non-additive
        noise. The distributions are zero-centered Gaussians with variance in
        the range [1,2].
        """
        for _ in range(self.num_nodes):
            var = np.random.uniform(low=1, high=2)
            u = lambda n, var=var: np.random.normal(loc=0, scale=np.sqrt(var), size=n)
            self.us.append(u)

    def sample(self, num_samples: int, do: int) -> np.ndarray:
        """
        Generates samples from an interventional distribution in the SCM.

        Args:
            num_samples: Number of samples to draw from the distribution
            do: Interventional distribution to sample from

        Returns:
            A num_samples x 2N+1 matrix. The last N+1 entries of each row
            comprise a one-hot vector indicating from which distribution the
            samples were drawn.
        """
        A = self.A.copy()
        if do >= 0:
            A[:, do] = 0
        X = np.zeros((num_samples, self.num_nodes))
        for i in range(self.num_nodes):
            ins = X[:, np.argwhere(A[:, i])].reshape(num_samples, -1)
            u = self.us[i](num_samples) if do != i else normal(loc=2, scale=1, size=num_samples)
            is_root = not ins.any()
            if self.fn_type == FnType.LINEAR:
                outs = u if is_root or do == i else self.fs[i](ins) + u
            elif self.fn_type == FnType.NONLINEAR:
                outs = u if is_root or do == i else self.fs[i](ins) + 0.4*u
            X[:, i] = outs.flatten()
        return X

    def make_dataset(self, samples_per_intervention: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Constructs a dataset containing samples from all interventional
        distributions.

        Args:
            samples_per_intervention: Number of samples drawn from each
            interventional distribution

        Returns:
            An N*samples_per_intervention x 2N+1 matrix as well as a vector of
            means and standard deviations of intervention values after
            normalization (used for mimicking interventions in the NCM). The
            last N+1 entries of each row in the matrix of samples comprise a
            one-hot vector indicating from which distribution the samples were
            drawn.
        """
        X = np.zeros((0, 2*self.num_nodes+1))
        for do in range(-1, self.num_nodes):
            X_do = self.sample(num_samples=samples_per_intervention, do=do)
            onehot = np.zeros((samples_per_intervention, self.num_nodes+1))
            onehot[:, do] = 1
            X_do = np.concatenate((X_do, onehot), axis=1)
            X = np.concatenate((X, X_do))
        means = np.mean(X[:, :self.num_nodes], axis=0)
        stds = np.std(X[:, :self.num_nodes], axis=0)

        X[:, :self.num_nodes] = (X[:, :self.num_nodes] - means) / stds

        means, stds = [], []
        for i in range(self.num_nodes):
            intervened_samples = X[X[:, self.num_nodes+i] == 1]
            means.append(np.mean(intervened_samples[:, i]))
            stds.append(np.std(intervened_samples[:, i]))
        np.random.shuffle(X)
        
        return X, means, stds
