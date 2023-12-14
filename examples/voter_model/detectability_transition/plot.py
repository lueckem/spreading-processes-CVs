import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.layout_engine import TightLayoutEngine
import matplotlib
import numpy as np
from sponet import load_params


def plot_tm():
    # plt.rcParams["font.size"] = 13

    num_params = 4
    xis = [np.load(f"data/xi_{i}.npy") for i in range(num_params)]
    p_outs = [0.001, 0.003, 0.005, 0.007]
    p_in = 0.1

    fig = plt.figure(figsize=(4, 4))
    # fig.suptitle("$p_{in} = 0.1$")

    for i, xi in enumerate(xis):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(xi[:, 0], xi[:, 1], xi[:, 2], c=-xi[:, 0])

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        custom_lim = (np.min(xi[:, 0]), np.max(xi[:, 0]))
        plt.setp(ax, xlim=custom_lim, ylim=custom_lim, zlim=custom_lim)

        ax.view_init(27, -126, 0)

        ax.set_title(r"$\frac{p_{out}}{p_{in}} =" + f" {(p_outs[i] / p_in):.2f}$")

    layout = TightLayoutEngine(pad=0.1)
    layout.execute(fig)


    fig.savefig(f"plots/plot_tm.pdf")
    plt.show()
