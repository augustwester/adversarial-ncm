import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from scm import SCM
from generator import Generator
from discriminator import Discriminator
from train import train
from graph import make_structured_graph, make_random_graph
from custom_types import GraphType, FnType
from datetime import datetime

def eval(graph_type, fn_type, num_nodes, batch_size, num_epochs, edge_probability=None):
    if graph_type is not GraphType.RANDOM:
        A = make_structured_graph(graph_type, num_nodes)
    elif args.edge_probability is not None:
        A = make_random_graph(edge_probability, num_nodes)

    scm = SCM(A, fn_type)
    X = torch.tensor(scm.make_dataset(samples_per_intervention=1000))
    g = Generator(num_nodes, num_dags=1, temperature=1)
    d = Discriminator(num_nodes)

    A_pred, g_losses, d_losses, p_hist = train(X, g, d, batch_size, num_epochs)
    shd = np.count_nonzero(A-A_pred)
    return g, scm, shd, g_losses, d_losses, p_hist

def save_plot(g, scm, g_losses, d_losses, p_hist, output_dir):
    fig, ax = plt.subplots(2, 1, figsize=(6,10))

    ax[0].set_title("Losses")
    ax[0].plot(g_losses, label="g_loss")
    ax[0].plot(d_losses, label="d_loss")
    ax[0].legend()

    ax[1].set_title("Edge beliefs")
    for r in range(g.num_nodes):
        for c in range(g.num_nodes):
            if r == c: continue
            if len(p_hist) > 9 and p_hist[r*g.num_nodes+c][-1] < 0.2: continue
            ax[1].plot(p_hist[r*g.num_nodes+c], label=f"x{r+1}->x{c+1}")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir + "plots.png")

    fig, ax = plt.subplots(g.num_nodes, g.num_nodes, figsize=(16,16))

    batch_size = 256
    z = torch.randn(batch_size, g.num_nodes)
    do = None

    X_g, A = g(z, do_idx=do)
    X_g = X_g.detach().numpy()
    X_data = scm.sample(batch_size, do=do)

    for i in range(g.num_nodes):
        for j in range(g.num_nodes):
            ax[i,j].scatter(X_g[:,i], X_g[:,j], c="r")
            ax[i,j].scatter(X_data[:,i], X_data[:,j], c="b")

    fig.tight_layout()
    fig.savefig(output_dir + "samples.png")

def save_txt(shd, args, output_dir):
    txt = open(output_dir + "info.txt", "w")
    txt.write(
        f"""
        Number of nodes: {args.num_nodes}
        Graph structure: {args.structure}
        Function type: {args.fn_type}
        Batch size: {args.batch_size}
        Number of epochs: {args.num_epochs}
        SHD: {shd}
        """)
    txt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--num_nodes", type=int, required=True)
parser.add_argument("--structure", type=str, required=True)
parser.add_argument("--fn_type", type=str, required=True)
parser.add_argument("--edge_probability", type=float, required=False)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_epochs", type=int, default=512)

args = parser.parse_args()

assert args.num_nodes > 1, "Minimum graph cardinality is 2"

graph_type = GraphType(args.structure.lower())
fn_type = FnType(args.fn_type.lower())

stats = eval(graph_type,
             fn_type,
             args.num_nodes,
             args.batch_size,
             args.num_epochs,
             args.edge_probability)
g, scm, shd, g_losses, d_losses, p_hist = stats

now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = f"./results/{now_str}/"
os.makedirs(output_dir)

save_plot(g, scm, g_losses, d_losses, p_hist, output_dir)
save_txt(shd, args, output_dir)
