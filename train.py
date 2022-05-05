import numpy as np
import torch
import torch.nn as nn
import random
from torch.optim import Adam, RMSprop
from tqdm import tqdm

def train(X, g, d, batch_size, num_epochs, g_lr, d_lr, edge_penalty=0.25, dag_penalty=1.0):
    g_opt = Adam(g.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_opt = Adam(d.parameters(), lr=d_lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    num_batches = X.shape[0] // batch_size
    P_hist = [[] for _ in range(g.num_nodes**2)]
    g_losses, d_losses = [], []

    random_idxs = list(range(g.num_nodes))
    random_idxs = [-1] + random_idxs

    for k in (pbar := tqdm(range(num_epochs))):
        random.shuffle(random_idxs)
        for i in random_idxs:
            for j in range(num_batches):
                # train generator (NCM)
                g_opt.zero_grad()
                z = torch.randn(batch_size, g.num_nodes)
                X_g, A = g(z, i if i >= 0 else None)
                preds = d(X_g)
                y_real = torch.ones(batch_size, 1)
                if k <= 0.1 * num_epochs:
                    g_loss = bce(preds, y_real)
                #elif k <= 0.75 * num_epochs:
                elif k <= 0.5 * num_epochs:
                    #g_loss = 10*bce(preds, y_real) + 0.5 * A.sum() / g.num_nodes
                    g_loss = bce(preds, y_real) + edge_penalty * A.sum() / g.num_nodes
                else:
                    m_exp = torch.trace(torch.matrix_exp(g.edge_beliefs.edge_beliefs)) - g.num_nodes
                    #g_loss = 10*bce(preds, y_real) + 2*m_exp
                    g_loss = bce(preds, y_real) + dag_penalty*m_exp
                g_loss.backward()
                g_opt.step()

                # train discriminator
                d_opt.zero_grad()
                z = torch.randn(batch_size, g.num_nodes)
                X_g, A = g(z, i if i >= 0 else None)
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
        #if k % 10 == 0:
        #    print(np.around(g.edge_beliefs.edge_beliefs.detach().numpy(), decimals=3))

    """
    i = -1
    for k in (pbar := tqdm(range(num_epochs))):
        for j in range(num_batches):
            # train generator (NCM)
            g_opt.zero_grad()
            z = torch.randn(batch_size, g.num_nodes)
            X_g, A = g(z, i if i >= 0 else None)
            preds = d(X_g)
            y_real = torch.ones(batch_size, 1)
            #if k <= 0.75 * num_epochs:
            #    g_loss = bce(preds, y_real) + 0.25 * A.sum() / g.num_nodes
            #else:
            #    m_exp = torch.trace(torch.matrix_exp(torch.sigmoid(g.edge_beliefs.P.T))) - g.num_nodes
            #    g_loss = bce(preds, y_real) + 0.5*m_exp
            m_exp = torch.trace(torch.matrix_exp(torch.sigmoid(g.edge_beliefs.P.T))) - g.num_nodes
            g_loss = bce(preds, y_real) + 0.05*(A.sum() / g.num_nodes) + (1 - 0.999**k)*m_exp
            g_loss.backward()
            g_opt.step()

            # train discriminator
            d_opt.zero_grad()
            z = torch.randn(batch_size, g.num_nodes)
            X_g, A = g(z, i if i >= 0 else None)
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

            #P = torch.sigmoid(g.edge_beliefs.P.T).detach().numpy()
            #entropies = -P * np.log(P)
            #mean_entropies = np.mean(entropies, axis=1)
            #i = np.argmax(mean_entropies)
            #i = np.unravel_index(entropies.argmax(), P.shape)[0]
            #i = np.random.choice([-1, i], p=[1/(g.num_nodes+1), 1-1/(g.num_nodes+1)])
            i = np.random.choice(list(range(-1, g.num_nodes)))

        #if k % 100 == 0:
        #    print(np.around(torch.sigmoid(g.edge_beliefs.P.T).detach().numpy(), decimals=3))
    """            
    P = torch.sigmoid(g.edge_beliefs.P.T).detach().numpy()
    A_pred = P > 0.6
    return A_pred, g_losses, d_losses, P_hist
