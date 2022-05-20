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
          num_pretraining_epochs=100,
          threshold=0.5):
    """
    Trains the NCM and matrix of edge beliefs represented by the generator.

    Args:
        X: Training data from the underlying ground truth SCM
        g: Generator containing the NCM and associated edge beliefs
        d: Discriminator network
        batch_size: The batch size used during training
        num_epochs: Number of epochs to train for (excluding pretraining)
        num_pretraining_epochs: Number of epochs for pretraining the model
        threshold: The threshold value by which an edge is included in the final prediction

    Returns:
        The predicted adjacency matrix, a list of recorded losses for the
        generator and discriminator, and a list of edge beliefs recorded
        throughout training.
    """

    g_opt = RMSprop(g.parameters(), lr=1e-3)
    d_opt = RMSprop(d.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    N = g.num_nodes
    P_hist = [[] for _ in range(N**2)]
    g_losses, d_losses = [], []

    def run_epoch(X, lambda_e, lambda_dag, do=None, update_edge_beliefs=True):
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
            if not update_edge_beliefs:
                g.edge_beliefs.P.grad = torch.zeros_like(g.edge_beliefs.P)
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

    print("Pretraining...")
    X_data = X[X[:,-1] == 1]
    for _ in (_ := tqdm(range(num_pretraining_epochs))):
        run_epoch(X_data, lambda_e=0, lambda_dag=0, do=-1, update_edge_beliefs=False)

    print("Training...")
    for i in (pbar := tqdm(range(num_epochs))):
        progress = i / num_epochs
        lambda_e = max(0.1 - (progress * 0.1), 0.01)
        lambda_dag = progress
        run_epoch(X, lambda_e, lambda_dag, do=None)

        for r in range(N):
            for c in range(N):
                belief = torch.sigmoid(g.edge_beliefs.P.T[r,c]).item()
                P_hist[r*N+c].append(belief)

        if i % 10 == 0:
            eb = g.edge_beliefs.edge_beliefs.detach().numpy()
            print(np.around(eb, decimals=3))

    P = torch.sigmoid(g.edge_beliefs.P.T).detach().numpy()
    A_hat = P > threshold
    return A_hat, g_losses, d_losses, P_hist
