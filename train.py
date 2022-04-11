import torch
import torch.nn as nn
import random
from torch.optim import Adam
from tqdm import tqdm

def train(X, g, d, batch_size=256, num_epochs=128):
    g_opt = Adam(g.parameters(), lr=1e-3, betas=(0.5, 0.999))
    d_opt = Adam(d.parameters(), lr=1e-3, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    num_batches = X.shape[0] // batch_size
    P_hist = [[] for _ in range(g.num_nodes**2)]
    g_losses, d_losses = [], []

    random_idxs = list(range(g.num_nodes))
    random_idxs = [-1] + random_idxs

    for k in (pbar := tqdm(range(num_epochs))):
        random.shuffle(random_idxs)
        for i in random_idxs:
            do_idx = i
            for j in range(num_batches):
                # train generator (NCM)
                g_opt.zero_grad()
                z = torch.randn(batch_size, g.num_nodes)
                X_g, A = g(z, do_idx if do_idx >= 0 else None)
                preds = d(X_g)
                y_real = torch.ones(batch_size, 1)
                if k <= 0.75 * num_epochs:
                    g_loss = bce(preds, y_real) + 0.5*A.mean()
                else:
                    m_exp = torch.trace(torch.matrix_exp(torch.sigmoid(g.edge_beliefs.P.T))) - g.num_nodes
                    g_loss = bce(preds, y_real) + 2*m_exp
                g_loss.backward()
                g_opt.step()

                # train discriminator
                d_opt.zero_grad()
                z = torch.randn(batch_size, g.num_nodes)
                X_g, A = g(z, do_idx if do_idx >= 0 else None)
                X_data = X[j*batch_size:j*batch_size+batch_size]
                X_all = torch.cat((X_g, X_data)).float()
                preds = d(X_all)
                y_fake = torch.zeros(batch_size, 1)
                y_real = torch.ones_like(y_fake)
                y = torch.cat((y_fake, y_real))
                d_loss = bce(preds, y)
                d_loss.backward()
                d_opt.step()

                for r in range(g.num_nodes):
                    for c in range(g.num_nodes):
                        belief = torch.sigmoid(g.edge_beliefs.P.T[r,c]).item()
                        P_hist[r*g.num_nodes+c].append(belief)
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
    P = torch.sigmoid(g.edge_beliefs.P.T).detach().numpy()
    A_pred = P > 0.6
    return A_pred, g_losses, d_losses, P_hist
