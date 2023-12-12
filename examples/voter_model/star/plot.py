import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.layout_engine import TightLayoutEngine
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from sponet import load_params
from sponet.collective_variables import OpinionShares


def plot_tm():
    plt.rcParams["font.size"] = 13

    xi = np.load("data/xi.npy")
    x_anchor = np.load("data/x_data.npz")["x_anchor"]
    cv = OpinionShares(2, True)
    c = cv(x_anchor)

    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(xi[:, 0], xi[:, 1], xi[:, 2], c=c[:, 0])

    ax.set_xlabel(r"$\varphi_1$", labelpad=-13)
    ax.set_ylabel(r"$\varphi_2$", labelpad=-13)
    ax.set_zlabel(r"$\varphi_3$", labelpad=-16)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    custom_lim = (np.min(xi[:, 0]), np.max(xi[:, 0]))
    plt.setp(ax, xlim=custom_lim, ylim=custom_lim, zlim=custom_lim)

    ax.view_init(50, -43, 0)

    layout = TightLayoutEngine(pad=-1.2)
    layout.execute(fig)
    fig.subplots_adjust(right=1.05)

    # plt.tight_layout()
    fig.savefig("plots/plot_tm.pdf")
    plt.show()


def plot_cv():
    plt.rcParams["font.size"] = 23
    caption_size = 28

    params = load_params("data/params.pkl")
    network = params.network
    pos = nx.kamada_kawai_layout(network)

    alphas = np.load("data/cv_optim.npz")["alphas"]
    # alphas1 /= np.max(np.abs(alphas1))
    # v1, v2 = -1, 1

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.25))

    axs = [axes[0], axes[1]]
    for i in range(2):
        ax = axs[i]
        al = alphas[:, i]
        img = nx.draw_networkx_nodes(
            network, pos=pos, ax=ax, node_color=al, node_size=80  # , vmin=v1, vmax=v2
        )
        nx.draw_networkx_edges(network, pos, ax=ax)
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_ticks([-1, 0, 1])

        cut = 1.2
        xmax = cut * max(xx for xx, yy in pos.values())
        ymax = cut * max(yy for xx, yy in pos.values())
        xmin = cut * min(xx for xx, yy in pos.values())
        ymin = cut * min(yy for xx, yy in pos.values())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    layout = TightLayoutEngine(pad=0.4)
    layout.execute(fig)

    fig.savefig(f"plots/plot_cv.pdf")
