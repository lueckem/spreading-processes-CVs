import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.layout_engine import TightLayoutEngine
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from cnvm.parameters import load_params


def plot_tm():
    plt.rcParams["font.size"] = 13

    xi = np.load("data/xi.npy")

    fig = plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot(projection="3d")

    scale_x = 1.5
    scale_y = 1.5
    scale_z = 1

    ax.scatter(xi[:, 0], xi[:, 2], xi[:, 1], c=-xi[:, 0])

    ax.set_xlabel(r"$\xi_3$", labelpad=-13)
    ax.set_ylabel(r"$\xi_1$", labelpad=-13)
    ax.set_zlabel(r"$\xi_2$", labelpad=-16)
    # ax.set_aspect('equal')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    custom_lim = (np.min(xi[:, 0]), np.max(xi[:, 0]))
    plt.setp(ax, xlim=custom_lim, ylim=custom_lim, zlim=custom_lim)

    ax.get_proj = lambda: np.dot(
        Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1])
    )
    # ax.text2D(-0.15, 0.85, "(a)", transform=ax.transAxes, fontsize=15)

    ax.view_init(13, -146, 0)

    layout = TightLayoutEngine(pad=-0.4)
    layout.execute(fig)
    fig.subplots_adjust(right=1.05)

    # plt.tight_layout()
    fig.savefig("plots/plot_tm.pdf")
    plt.show()


def plot_cv():
    plt.rcParams["font.size"] = 23
    caption_size = 28

    xi = np.load("data/xi.npy")
    params = load_params("data/params.pkl")
    network = params.network
    pos = nx.spring_layout(network, seed=100, k=0.09)
    alphas = np.load("data/cv_optim_degree_weighted.npz")["alphas"]
    colors = np.load("data/cv_optim_degree_weighted.npz")["xi_fit"]

    xi /= np.max(np.abs(xi))
    colors /= np.max(np.abs(colors))
    v_min, v_max = -1, 1

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

    # figure 1
    ax = axes[0, 0]
    ax.plot(xi[:, 0], colors[:, 0], "o")
    ax.set_ylabel(r"$\bar{\varphi}_1$")
    ax.set_xlabel(r"$\varphi_1$")
    ax.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, va="top", fontsize=caption_size)

    # figure 2-4
    captions = ["(b)", "(c)", "(d)"]
    labels = [r"$\Lambda_{1}$", r"$\Lambda_{2}$", r"$\Lambda_{3}$"]
    axs = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for i in range(3):
        ax = axs[i]
        this_alphas = alphas[:, i] / np.max(np.abs(alphas[:, i]))
        img = nx.draw_networkx_nodes(
            network,
            pos=pos,
            ax=ax,
            node_color=this_alphas,
            node_size=50,
            vmin=v_min,
            vmax=v_max,
        )
        nx.draw_networkx_edges(network, pos, ax=ax)
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_ticks([-1, 0, 1])

        cut = 1.05
        xmax = cut * max(xx for xx, yy in pos.values())
        ymax = cut * max(yy for xx, yy in pos.values())
        xmin = cut * min(xx for xx, yy in pos.values())
        ymin = cut * min(yy for xx, yy in pos.values())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(labels[i])
        ax.text(
            0.05,
            0.95,
            captions[i],
            transform=ax.transAxes,
            va="top",
            fontsize=caption_size,
        )

    layout = TightLayoutEngine(pad=0)
    layout.execute(fig)

    # make plot [0,0] smaller
    position = axes[0, 0].get_position()
    new_position = rescale_bbox(position, 0.8)
    axes[0, 0].set_position(new_position)

    fig.savefig(f"plots/plot_cv.pdf")


def rescale_bbox(
    bbox: Bbox,
    factor: float,
) -> Bbox:
    new_width = factor * bbox.width
    x0, x1 = bbox.intervalx[0], bbox.intervalx[0] + new_width
    y0, y1 = bbox.intervaly[0], bbox.intervaly[1]
    return Bbox(np.array([[x0, y0], [x1, y1]]))


# def plot_cv():
#     plt.rcParams["font.size"] = 23
#     caption_size = 28
#
#     params = load_params("data/params.pkl")
#     network = params.network
#     pos = nx.kamada_kawai_layout(network)
#
#     alphas1 = np.load("data/cv_optim.npz")["alphas"]
#     alphas1 /= np.max(np.abs(alphas1))
#
#     alphas2 = np.load("data/cv_optim_degree_weighted.npz")["alphas"]
#     alphas2 /= np.max(np.abs(alphas2))
#
#     alphas = [alphas1, alphas2]
#     v1, v2 = -1, 1
#
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.25))
#
#     captions = ["(a)", "(b)"]
#     axs = [axes[0], axes[1]]
#     for i in range(2):
#         ax = axs[i]
#         al = alphas[i][:, 0]
#         img = nx.draw_networkx_nodes(
#             network, pos=pos, ax=ax, node_color=al, node_size=80, vmin=v1, vmax=v2
#         )
#         nx.draw_networkx_edges(network, pos, ax=ax)
#         cbar = fig.colorbar(img, ax=ax)
#         cbar.set_ticks([-1, 0, 1])
#
#         cut = 1.2
#         xmax = cut * max(xx for xx, yy in pos.values())
#         ymax = cut * max(yy for xx, yy in pos.values())
#         xmin = cut * min(xx for xx, yy in pos.values())
#         ymin = cut * min(yy for xx, yy in pos.values())
#         ax.set_xlim(xmin, xmax)
#         ax.set_ylim(ymin, ymax)
#         ax.text(
#             0.05,
#             0.97,
#             captions[i],
#             transform=ax.transAxes,
#             va="top",
#             fontsize=caption_size,
#         )
#
#     layout = TightLayoutEngine(pad=0.4)
#     layout.execute(fig)
#
#     fig.savefig(f"plots/plot_cv.pdf")