from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.layout_engine import TightLayoutEngine


def plot_tm():
    plt.rcParams["font.size"] = 13

    xi = np.load("data/xi.npy")

    fig = plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot(projection="3d")

    scale_x = 1.5
    scale_y = 1.5
    scale_z = 1

    ax.scatter(xi[:, 0], xi[:, 2], xi[:, 1], c=-xi[:, 0])

    ax.set_xlabel(r"$\varphi_1$", labelpad=-13)
    ax.set_ylabel(r"$\varphi_2$", labelpad=-13)
    ax.set_zlabel(r"$\varphi_3$", labelpad=-16)
    # ax.set_aspect('equal')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    custom_lim = (np.min(xi[:, 0]), np.max(xi[:, 0]))
    custom_lim2 = (0.9 * np.min(xi[:, 0]), 0.9 * np.max(xi[:, 0]))
    plt.setp(ax, xlim=custom_lim, ylim=custom_lim2, zlim=custom_lim2)

    ax.get_proj = lambda: np.dot(
        Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1])
    )
    # ax.text2D(-0.15, 0.85, "(a)", transform=ax.transAxes, fontsize=15)

    ax.view_init(15, -127, 0)

    layout = TightLayoutEngine(pad=-0.4)
    layout.execute(fig)
    fig.subplots_adjust(right=1.05)

    plt.savefig("plots/plot_tm.pdf")
    plt.show()
