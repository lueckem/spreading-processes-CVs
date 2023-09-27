import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.layout_engine import TightLayoutEngine
import numpy as np
from sponet import load_params


def plot3d():
    xi = np.load("data/xi.npy")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(xi[:, 0], xi[:, 2], xi[:, 1], c=xi[:, 2])

    ax.set_xlabel(r"$\xi_1$")
    ax.set_ylabel(r"$\xi_3$")
    ax.set_zlabel(r"$\xi_2$")

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    ax.view_init(14, -47, 0)
    plt.show()
    plt.tight_layout()
    plt.savefig("plots/plot_tm.pdf")
