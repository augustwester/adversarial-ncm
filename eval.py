import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from custom_types import GraphType, FnType
from graph import make_graph
from scm import SCM
from generator import Generator
from discriminator import Discriminator
from train import train
from run import compute_shd, run, save_loss_plot, save_samples_plot, save_txt

def evaluate(graph_type: GraphType, fn_type: FnType, num_nodes: int, batch_size: int):
    """
    Evaluate the proposed method 5 times on data from SCMs obeying the
    specified parameters.

    Args:
        graph_type: The type of graph (e.g. chain or ER-1)
        fn_type: Specifies whether the functional relationships should be linear or nonlinear
        num_nodes: The number of nodes in the graph
        batch_size: The batch size used during training

    Returns:
        No return value. Saves loss plot and edge belief plot to disk along
        with a text file containing relevant information on the run.
    """
    container_dir = f"./eval/{num_nodes}/{graph_type.name}"
    os.makedirs(container_dir, exist_ok=True)

    shds = []
    for i in range(5):
        stats = run(graph_type,
                    fn_type,
                    num_nodes,
                    batch_size=batch_size,
                    num_epochs=None)
        g, scm, shd, g_losses, d_losses, p_hist = stats
        shds.append(shd)
        output_dir = "/".join([container_dir, str(i+1)]) + "/"
        os.makedirs(output_dir)
        graph_name = graph_type.name.lower() + str(num_nodes)
        save_loss_plot(graph_name, g_losses, d_losses, p_hist, output_dir)
        txt = open(output_dir + "/info.txt", "w")
        txt.write(
            f"""
            Number of nodes: {num_nodes}
            Graph structure: {graph_type.name}
            SHD: {shd}""")
        txt.close()

    mean_shd = np.mean(shds)
    std = np.std(shds)
    txt = open("/".join([container_dir, "summary.txt"]), "w")
    txt.write(
        f"""
        Number of nodes: {num_nodes}
        Graph structure: {graph_type.name}
        Mean SHD: {mean_shd}
        Standard deviation: {std}
        """)
    txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--type", type=str, required=True)

    args = parser.parse_args()
    fn_type = FnType(args.type.lower())
    batch_size = args.batch_size

    graphs = [GraphType.CHAIN,
              GraphType.COLLIDER,
              GraphType.BIDIAG,
              GraphType.TREE,
              GraphType.JUNGLE,
              GraphType.ER1,
              GraphType.ER2,
              GraphType.FULL]
    #nodes = [4, 6, 8, 10]
    nodes = [6,8,10]

    for num_nodes in nodes:
        ps = [Process(target=evaluate, args=(graph, fn_type, num_nodes, batch_size)) for graph in graphs]
        for p in ps: p.start()
        for p in ps: p.join()
