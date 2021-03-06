import numpy as np
import torch
import random
from torch import nn, Tensor
from torch.optim import Adam, RMSprop
from generator import Generator
from discriminator import Discriminator
from tqdm import tqdm

def train(X: Tensor,
          g: Generator,
          d: Discriminator,
          batch_size: int,
          num_epochs: int,
          early_stopping=True,
          threshold=0.2,
          weight_decay=0.0,
          verbose=False):
    """
    Trains the NCM and matrix of edge beliefs represented by the generator.

    Args:
        X: Training data from the underlying ground truth SCM
        g: Generator containing the NCM and associated edge beliefs
        d: Discriminator network
        batch_size: The batch size used during training
        num_epochs: Number of epochs to train for
        early_stopping: Whether or not to stop training once convergence is detected
        threshold: The threshold value by which an edge is included in the final prediction
        weight_decay: Coefficient on weight decay (L2 regularization) on the parameters of the model's MLPs
        verbose: If set to true, the current matrix of edge beliefs will be printed every 10 epochs. Default: False.

    Returns:
        The predicted adjacency matrix, a list of recorded losses for the
        generator and discriminator, and a list of edge beliefs recorded
        throughout training.
    """
    e_opt = RMSprop(g.edge_beliefs.parameters(), lr=5e-3)
    n_opt = RMSprop(g.ncm.parameters(), lr=1e-3, weight_decay=weight_decay)
    d_opt = RMSprop(d.parameters(), lr=1e-3 if weight_decay == 0 else 1e-4, weight_decay=weight_decay)
    bce = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.LinearLR(e_opt, start_factor=0.1, total_iters=num_epochs)

    N = g.num_nodes
    P_hist = [[] for _ in range(N**2)]
    g_losses, d_losses = [], []

    def run_epoch(X, lambda_e, lambda_dag, g_opt, d_opt, do=None):
        X = X[torch.randperm(len(X))]
        num_batches = len(X) // batch_size
        batch_sizes = [batch_size for _ in range(num_batches)]

        if len(X) % batch_size > 0:
            batch_sizes.append(len(X) - num_batches*batch_size)

        for j, bs in enumerate(batch_sizes):
            # train generator (NCM)
            g_opt.zero_grad()
            z = torch.rand(bs, N)
            A, order = g.edge_beliefs.sample_dags(bs)
            X_g = g(z, A, order, do=do)
            preds = d(X_g)
            y_real = torch.ones(bs, 1)
            m_exp = torch.trace(torch.matrix_exp(g.edge_beliefs.edge_beliefs)) - N
            g_loss = bce(preds, y_real) + lambda_e * A.mean(0).sum() + lambda_dag * m_exp
            g_loss.backward()
            g_opt.step()

            # train discriminator
            d_opt.zero_grad()
            z = torch.rand(bs, N)
            A, order = g.edge_beliefs.sample_dags(bs)
            X_g = g(z, A, order, do=do)
            X_data = X[sum(batch_sizes[:j]):sum(batch_sizes[:j])+bs]
            X_all = torch.cat((X_g, X_data)).float()
            preds = d(X_all)
            y_fake = torch.zeros(bs, 1)
            y_real = torch.ones_like(y_fake)
            y = torch.cat((y_fake, y_real))
            d_loss = bce(preds, y)
            d_loss.backward()
            d_opt.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

    for i in (_ := tqdm(range(num_epochs))):
        progress = (i+1) / num_epochs
        lambda_e = max(0.1 - (progress * 0.1), 0.01)
        lambda_dag = 0.1*progress

        run_epoch(X, 0, 0, n_opt, d_opt)
        run_epoch(X, lambda_e, lambda_dag, e_opt, d_opt)
        scheduler.step()

        for r in range(N):
            for c in range(N):
                belief = torch.sigmoid(g.edge_beliefs.P.T[r,c]).item()
                P_hist[r*N+c].append(belief)

        if verbose and i % 10 == 0:
            print(np.around(g.edge_beliefs.edge_beliefs.detach().numpy(), decimals=3))

        if early_stopping:
            A = g.edge_beliefs.edge_beliefs
            num_pos = len(A[A > threshold])
            num_neg = len(A[A < 0.01])
            if num_neg > N and num_pos+num_neg == N**2:
                break

    P = torch.sigmoid(g.edge_beliefs.P.T).detach().numpy()
    A_hat = P > threshold
    return A_hat, g_losses, d_losses, P_hist
