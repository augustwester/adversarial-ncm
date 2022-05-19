import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scm import SCM
from generator import Generator
from discriminator import Discriminator
from train import train
from graph import make_graph
from custom_types import GraphType, FnType
from datetime import datetime

sns.set_theme()

def compute_shd(A, A_hat):
    """
    Compute the structual Hamming distance (SHD) between two graphs with
    adjacency matrices A and A_hat. This implementation assigns an SHD of 1 if
    an edge flip is sufficient to recover the true graph.

    Code taken from: https://github.com/ElementAI/causal_discovery_toolbox

    Args:
        A: The first adjacency matrix
        A_hat: The second adjacency matrix

    Returns:
        A positive integer
    """
    diff = np.abs(A - A_hat)
    diff = diff + diff.T
    diff[diff > 1] = 1
    return np.sum(diff) / 2

def eval(graph_type, fn_type, num_nodes, batch_size, num_epochs):
    """
    Evaluate the proposed method on data sampled from an SCM. The SCM is
    constructed based on the specified graph type, function type, and number
    of nodes. The data contains 1000 samples for each distribution.

    Returns the trained NCM, SHD, and a history of losses and edge beliefs.
    """
    A = make_graph(graph_type, num_nodes)
    scm = SCM(A, fn_type)
    X = torch.tensor(scm.make_dataset(samples_per_intervention=1000))
    g = Generator(num_nodes, num_dags=1, temperature=0.1)
    d = Discriminator(num_nodes)
    num_epochs = num_nodes*350 if num_epochs is None else num_epochs
    A_pred, g_losses, d_losses, p_hist = train(X, g, d, batch_size, num_epochs)
    shd = compute_shd(A, A_pred)
    return g, scm, shd, g_losses, d_losses, p_hist

def save_loss_plot(graph_name, g_losses, d_losses, p_hist, output_dir):
    """
    Plot losses of generator (NCM) and discriminator and save to disk.
    """
    N = int(np.sqrt(len(p_hist)))
    fig, ax = plt.subplots(2, 1, figsize=(6,10))

    ax[0].set_title("Losses")
    ax[0].plot(g_losses, label="g_loss")
    ax[0].plot(d_losses, label="d_loss")
    ax[0].legend()

    ax[1].set_title(graph_name)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel(r"$\sigma(\gamma_{ij})$")

    for r in range(N):
        for c in range(N):
            if r == c: continue
            if len(p_hist) > 9 and p_hist[r*N+c][-1] < 0.2: continue
            ax[1].plot(p_hist[r*N+c], label=f"x{r+1}->x{c+1}")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir + "plots.png")

def save_samples_plot(g, scm, output_dir):
    dos = list(range(-1, g.num_nodes))

    for do in dos:
        fig, ax = plt.subplots(g.num_nodes, g.num_nodes, figsize=(16,16))
        batch_size = 100
        z = torch.rand(batch_size, g.num_nodes)

        A = (g.edge_beliefs.P.T > 0.6).int()
        while True:
            A_, order = g.edge_beliefs.sample_dags(1)
            A_ = A_[0]
            A_ = (A > 0.6).int()
            if (A == A_).all().item(): break

        A = A.repeat(batch_size, 1, 1)
        order = order.repeat(batch_size, 1)

        X_g = g(z, do_idx=do, A=A, order=order)
        X_g = X_g.detach().numpy()
        X_data = scm.sample(batch_size, do=do)

        for i in range(g.num_nodes):
            for j in range(g.num_nodes):
                ax[i,j].scatter(X_g[:,i], X_g[:,j], c="r")
                ax[i,j].scatter(X_data[:,i], X_data[:,j], c="b")

        fig.tight_layout()
        fig.savefig(output_dir + f"do-{do}.png")

def save_txt(shd, args, output_dir):
    txt = open(output_dir + "info.txt", "w")
    txt.write(
        f"""
        Number of nodes: {args.num_nodes}
        Graph structure: {args.structure}
        Function type: {args.fn_type}
        Batch size: {args.batch_size}
        Number of epochs: {350*args.num_nodes if args.num_epochs is None else args.num_epochs}
        SHD: {shd}
        """)
    txt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--structure", type=str, required=True)
    parser.add_argument("--fn_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=None)

    args = parser.parse_args()
    assert args.num_nodes > 1, "Minimum graph cardinality is 2"

    graph_type = GraphType(args.structure.lower())
    fn_type = FnType(args.fn_type.lower())

    stats = eval(graph_type,
                 fn_type,
                 args.num_nodes,
                 args.batch_size,
                 args.num_epochs)
    g, scm, shd, g_losses, d_losses, p_hist = stats

    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f"./results/{now_str}/"
    os.makedirs(output_dir)

    graph_name = graph_type.name.lower() + str(args.num_nodes)
    save_loss_plot(graph_name, g_losses, d_losses, p_hist, output_dir)
    save_samples_plot(g, scm, output_dir)
    save_txt(shd, args, output_dir)
