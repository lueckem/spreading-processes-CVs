from interpretable_cvs.validation import (
    sample_state_like_x,
    plot_x_levelset,
    mmd_from_trajs,
    plot_validation_mmd,
)
from sponet import load_params, sample_many_runs
import interpretable_cvs as ct
import pickle
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def build_level_sets():
    params = load_params("data/params.pkl")
    with open("data/cv.pkl", "rb") as file:
        cv = pickle.load(file)

    for c in cv.collective_variables:
        c.normalize = False

    # create state with "cluster" of 1's
    x1 = ct.create_anchor_points_local_clusters(
        params.network, params.num_opinions, 1, 1
    )[0, :]

    # randomly shuffle x1
    x2 = np.copy(x1)
    np.random.shuffle(x2)

    # different share of 1's
    x3 = np.random.randint(0, 2, params.num_agents)
    print(f"x0: {np.mean(x3)}, x1: {np.mean(x1)}")

    x = np.array([x1, x2, x3])
    np.save("data/x_levelset.npy", x)


def build_level_sets_2():
    params = load_params("data/params.pkl")
    coordinates = [0, 4, 5]
    x_anchor = np.load("data/x_data.npz")["x_anchor"]
    xi = np.load("data/xi.npy")[:, coordinates]

    # find points where xi_1 is close but the other coordinates are
    # as far apart as possible
    best_i, best_j = find_first_close_others_far(xi)
    x1 = x_anchor[best_i]
    x3 = x_anchor[best_j]

    # find point that is close to xi[i] but has entries as different as possible
    best_k = find_close_xi_different_x(xi, x_anchor, best_i)
    x2 = x_anchor[best_k]

    print(xi[best_i], xi[best_k], xi[best_j])
    print(f"x1: {np.mean(x1)}, x2: {np.mean(x2)}, x3: {np.mean(x3)}")

    x = np.array([x1, x2, x3])
    np.save("data/x_levelset.npy", x)


def find_first_close_others_far(xi):
    # find points where xi_1 is close but the other coordinates are
    # as far apart as possible.
    def first_close_others_far(p1, p2, abs_tol):
        if np.abs(p1[0] - p2[0]) < abs_tol:
            return np.linalg.norm(p1[1:] - p2[1:])
        return 0

    tol = (np.max(xi[:, 0]) - np.min(xi[:, 0])) / 100
    best_i, best_j = None, None
    dist = 0
    for i in range(xi.shape[0]):
        for j in range(xi.shape[0]):
            this_dist = first_close_others_far(xi[i, :], xi[j, :], tol)
            if this_dist > dist:
                dist = this_dist
                best_i, best_j = i, j

    return best_i, best_j


def find_close_xi_different_x(xi, x_anchor, i):
    # find point that is close to xi[i] but has entries as different as possible
    def close_xi_different_x(xi_1, xi_2, x_1, x_2, abs_tol):
        if np.linalg.norm(xi_1 - xi_2) < abs_tol:
            return np.linalg.norm(x_1 - x_2)
        return 0

    dist = 0
    best_k = None
    tol = max_norm_rows(xi) / 12
    for k in range(xi.shape[0]):
        this_dist = close_xi_different_x(xi[i, :], xi[k, :], x_anchor[i, :], x_anchor[k, :], tol)
        if this_dist > dist:
            dist = this_dist
            best_k = k
    if best_k is None:
        raise RuntimeError("Could not find k.")

    return best_k


def max_norm_rows(x: np.ndarray):
    """ Compute the maximum of the 2-norm of (r1 - r2),
     where r1 and r2 are rows of x"""
    dist = 0
    for i in range(x.shape[0]):
        for j in range(i):
            this_dist = np.linalg.norm(x[i] - x[j])
            if this_dist > dist:
                dist = this_dist
    return dist


def validate_mmd():
    print("Validating...")
    t_max = 200
    num_samples = 1000
    num_timesteps = 100

    params = load_params("data/params.pkl")
    with open("data/cv.pkl", "rb") as file:
        cv = pickle.load(file)

    x_init = np.load("data/x_levelset.npy")[:3]
    print("CV values of the three states:")
    print(cv(x_init))

    t, x = sample_many_runs(
        params,
        x_init,
        t_max,
        num_timesteps,
        num_runs=num_samples,
        n_jobs=-1,
    )

    c = np.zeros((3, num_samples, num_timesteps, cv.dimension))
    for i in [0, 1, 2]:
        for j in range(num_samples):
            c[i, j] = cv(x[i, j])

    np.savez_compressed("data/data_validate_full.npz", t=t, c=x)
    np.savez_compressed("data/data_validate.npz", t=t, c=c)

    mmd = mmd_from_trajs(x)
    np.save("data/mmd_validate_full.npy", mmd)
    mmd = mmd_from_trajs(c)
    np.save("data/mmd_validate.npy", mmd)


def plot_mmd():
    mmd = np.load("data/mmd_validate.npy")
    mmd_full = np.load("data/mmd_validate_full.npy")
    t = np.load("data/data_validate.npz")["t"]

    # fig, axes = plt.subplots(2, 1, figsize=(3.5, 6), sharex=True)
    # plot_validation_mmd(t, mmd, axes[0], r"MMD$_\varphi$")
    # plot_validation_mmd(t, mmd_full, axes[1], r"MMD")

    fig, axes = plt.subplots(figsize=(3.5, 3))
    plot_validation_mmd(t, mmd_full, axes, r"MMD")

    fig.tight_layout()
    fig.savefig("plots/validate_mmd.pdf")


def plot_level_set():
    x = np.load("data/x_levelset.npy")
    network = load_params("data/params.pkl").network
    fig = plot_x_levelset(x, network)
    fig.tight_layout()
    fig.savefig("plots/level_set.pdf")
