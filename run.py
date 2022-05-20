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

def compute_shd(A: np.ndarray, A_hat: np.ndarray) -> int:
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

def run(graph_type: GraphType,
        fn_type: FnType,
        num_nodes: int,
        batch_size: int,
        num_epochs: int) -> (Generator, SCM, list, list, list):
    """
    Evaluate the proposed method on data sampled from an SCM. The SCM is
    constructed based on the specified graph type, function type, and number
    of nodes. The data contains 1000 samples for each distribution.

    Args:
        graph_type: The type of graph (e.g. chain or ER-1)
        fun_type: Specifies whether the functional relationships should be linear or nonlinear
        num_nodes: The number of nodes in the graph
        batch_size: The batch size used during training
        num_epochs: Number of epochs to train for (excluding pretraining)

    Returns:
        The trained generator, SHD, and a history of losses and edge beliefs.
    """
    A = make_graph(graph_type, num_nodes)
    scm = SCM(A, fn_type)
    X = torch.tensor(scm.make_dataset(samples_per_intervention=1000))
    g = Generator(num_nodes, temperature=0.1)
    d = Discriminator(num_nodes)
    num_epochs = num_nodes*350 if num_epochs is None else num_epochs
    A_pred, g_losses, d_losses, p_hist = train(X, g, d, batch_size, num_epochs)
    shd = compute_shd(A, A_pred)
    return g, scm, shd, g_losses, d_losses, p_hist

def save_loss_plot(graph_name: str,
                   g_losses: list,
                   d_losses: list,
                   p_hist: list,
                   output_dir: str):
    """
    Plot losses of generator (NCM) and discriminator and save to disk.

    Args:
        graph_name: Name of the graph (used as title for the plot)
        g_losses: A list of generator losses throughout training
        d_losses: A list of discriminator losses throughout training
        p_hist: A list of edge beliefs during training
        output_dir: The directory to save the plot to
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

def save_samples_plot(g: Generator, scm: SCM, output_dir: str):
    """
    Plot samples from the SCM and samples from the trained NCM for comparison.

    Args:
        g: The generator (containing the NCM)
        scm: The SCM on which the generator / NCM was trained
        output_dir: The directory to save the plot to
    """
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

        X_g = g(z, A, order, do)
        X_g = X_g.detach().numpy()
        X_data = scm.sample(batch_size, do=do)

        for i in range(g.num_nodes):
            for j in range(g.num_nodes):
                ax[i,j].scatter(X_g[:,i], X_g[:,j], c="r")
                ax[i,j].scatter(X_data[:,i], X_data[:,j], c="b")

        fig.tight_layout()
        fig.savefig(output_dir + f"do-{do}.png")

def save_txt(shd: int, args: dict, output_dir: str):
    """
    Saves a text file containing a summary of the model / training / result.

    Args:
        shd: The final SHD after training
        args: A dictionary of command line arguments from argparse
        output_dir: The directory to save the file to
    """
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

    stats = run(graph_type,
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
