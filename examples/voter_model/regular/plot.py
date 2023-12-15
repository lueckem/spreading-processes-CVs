from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.layout_engine import TightLayoutEngine
from sponet import load_params
import networkx as nx

def plot_tm():
    plt.rcParams["font.size"] = 13

    xi = np.load("data/xi.npy")

    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot(projection="3d")

    # scale_x = 1.5
    # scale_y = 1.5
    # scale_z = 1

    indices = [0, 1, 2]

    ax.scatter(xi[:, indices[0]], xi[:, indices[1]], xi[:, indices[2]], c=-xi[:, 0])

    ax.set_xlabel(rf"$\varphi_{indices[0] + 1}$", labelpad=-13)
    ax.set_ylabel(rf"$\varphi_{indices[1] + 1}$", labelpad=-13)
    ax.set_zlabel(rf"$\varphi_{indices[2] + 1}$", labelpad=-16)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    custom_lim = (np.min(xi[:, 0]), np.max(xi[:, 0]))
    plt.setp(ax, xlim=custom_lim, ylim=custom_lim, zlim=custom_lim)

    # ax.get_proj = lambda: np.dot(
    #     Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1])
    # )
    # ax.text2D(-0.15, 0.85, "(a)", transform=ax.transAxes, fontsize=15)

    # ax.view_init(16, -115, 0)
    ax.view_init(23, 153, 0)

    layout = TightLayoutEngine(pad=-1.2)
    layout.execute(fig)
    fig.subplots_adjust(right=1.05)

    # plt.tight_layout()
    fig.savefig("plots/plot_tm.pdf")
    plt.show()


def plot_dimension_estimation():
    data = np.load("data/avg_sim.npz")
    s = data["s"]
    epsilons = data["epsilons"]
    dist_mat = np.load("data/dist_mat.npy")

    derivative = _central_differences(np.log(epsilons), np.log(s))

    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot()
    ax.loglog(epsilons, s, label=r"$S(\varepsilon)$")
    ax.loglog(epsilons, derivative, label=r"$\frac{d \log S(\varepsilon)}{d \log \varepsilon}$")
    ax.grid()
    ax.legend()
    ax.set_ylim((1 / dist_mat.shape[0] / 2, np.max(derivative) * 2))

    fig.savefig("plots/dimension_estimation.pdf")
    plt.show()


def _central_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute dy/dx via central differences.
    """
    out = np.zeros(len(x))
    for i in range(len(x)):
        upper_idx = min(i + 1, len(x) - 1)
        lower_idx = max(i - 1, 0)
        out[i] = (y[upper_idx] - y[lower_idx]) / (x[upper_idx] - x[lower_idx])
    return out


def plot_cv():
    coordinates = [0, 4, 5]

    xi = np.load("data/xi.npy")
    params = load_params("data/params.pkl")
    network = params.network
    pos = nx.spring_layout(network, seed=100, k=0.09)
    alphas = np.load("data/cv_optim.npz")["alphas"]
    xi_fit = np.load("data/cv_optim.npz")["xi_fit"]

    # xi /= np.max(np.abs(xi))
    # colors /= np.max(np.abs(colors))
    # v_min, v_max = -1, 1

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6, 6))
    left_axs = [axes[0, 0], axes[1, 0], axes[2, 0]]
    right_axs = [axes[0, 1], axes[1, 1], axes[2, 1]]

    for i in range(3):
        ax = left_axs[i]
        ax.plot(xi[:, coordinates[i]], xi_fit[:, i], "x")

    for i in range(3):
        ax = right_axs[i]
        this_alphas = alphas[:, i]
        img = nx.draw_networkx_nodes(
            network,
            pos=pos,
            ax=ax,
            node_color=this_alphas,
            node_size=50,
        )
        nx.draw_networkx_edges(network, pos, ax=ax)
        fig.colorbar(img, ax=ax)

    plt.show()
