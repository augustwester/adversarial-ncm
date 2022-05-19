import numpy as np
import torch
import torch.nn as nn
import random
from torch.optim import Adam, RMSprop
from tqdm import tqdm

def train(X, g, d, batch_size, num_epochs, threshold=0.5):
    g_opt = RMSprop(g.parameters(), lr=1e-3)
    d_opt = RMSprop(d.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    N = g.num_nodes
    num_batches = X.shape[0] // batch_size
    P_hist = [[] for _ in range(N**2)]
    g_losses, d_losses = [], []
    random_idxs = list(range(-1, N))

    def run_epoch(X, lambda_e, lambda_dag, do=None, update_edge_beliefs=True):
        num_batches = X.shape[0] // batch_size
        for j in range(num_batches):
            # train generator (NCM)
            g_opt.zero_grad()
            z = torch.rand(batch_size, N)
            A, order = g.edge_beliefs.sample_dags(batch_size)
            #A, order = g.edge_beliefs.sample_dags(1)
            #A = A.repeat(batch_size, 1, 1)
            #order = order.repeat(batch_size, 1)
            X_g = g(z, A, order, do_idx=do)
            preds = d(X_g)
            y_real = torch.ones(batch_size, 1)
            m_exp = torch.trace(torch.matrix_exp(g.edge_beliefs.edge_beliefs)) - N
            g_loss = bce(preds, y_real) + lambda_e * A.mean(0).sum() + lambda_dag * m_exp
            g_loss.backward()
            if not update_edge_beliefs:
                g.edge_beliefs.P.grad = torch.zeros_like(g.edge_beliefs.P)
            g_opt.step()

            # train discriminator
            d_opt.zero_grad()
            z = torch.rand(batch_size, N)
            A, order = g.edge_beliefs.sample_dags(batch_size)
            X_g = g(z, A, order, do_idx=do)
            X_data = X[j*batch_size:j*batch_size+batch_size]
            X_all = torch.cat((X_g, X_data)).float()
            preds = d(X_all)
            y_fake = torch.zeros(batch_size, 1)
            y_real = torch.ones_like(y_fake)
            y = torch.cat((y_fake, y_real))
            d_loss = bce(preds, y)
            d_loss.backward()
            d_opt.step()

    for i in (pbar := tqdm(range(num_epochs))):
        if i == 0:
            print("Pretraining...")
            X_data = X[X[:,-1] == 1]
            for _ in (_ := tqdm(range(100))):
                run_epoch(X_data, lambda_e=0, lambda_dag=0, do=-1, update_edge_beliefs=False)

        progress = i / (N*200)
        lambda_e = max(0.1 - (progress * 0.1), 0.01)
        lambda_dag = progress
        run_epoch(X, lambda_e, lambda_dag, do=None, update_edge_beliefs=True)

        for r in range(N):
            for c in range(N):
                belief = torch.sigmoid(g.edge_beliefs.P.T[r,c]).item()
                P_hist[r*N+c].append(belief)

        if i % 10 == 0:
            print(g.edge_beliefs.edge_beliefs)

    P = torch.sigmoid(g.edge_beliefs.P.T).detach().numpy()
    A_pred = P > threshold
    return A_pred, g_losses, d_losses, P_hist
