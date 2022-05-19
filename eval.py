import os
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

def run(graph_type, num_nodes):
    """
    Evaluate the proposed method 10 times on data from SCMs obeying the
    specified parameters.

    Saves loss plot and edge belief plot to disk along with a text file
    containing relevant information on the run.
    """
    container_dir = f"./parallel/{num_nodes}/{graph_type.name}"
    os.makedirs(container_dir, exist_ok=True)

    shds = []
    for i in range(5):
        stats = run(graph_type,
                    FnType.LINEAR,
                    num_nodes,
                    batch_size=256,
                    num_epochs=350*num_nodes)
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
    graphs = [GraphType.CHAIN,
              GraphType.COLLIDER,
              GraphType.BIDIAG,
              GraphType.TREE,
              GraphType.JUNGLE,
              GraphType.FULL]
    #nodes = [4, 6, 8, 10]
    nodes = [6, 8]

    for num_nodes in nodes:
        ps = []
        for graph in graphs:
            p = Process(target=run, args=(graph, num_nodes))
            p.start()
            ps.append(p)
        for p in ps: p.join()
