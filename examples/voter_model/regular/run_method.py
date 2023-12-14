import pickle
from sponet import CNVMParameters, save_params, load_params
import sponet.network_generator as ng
import numpy as np
import networkx as nx

from interpretable_cvs import (
    TransitionManifold,
    optimize_fused_lasso,
    sample_cnvm,
    build_cv_from_alpha,
)
import interpretable_cvs as ct


def setup_params():
    num_opinions = 2
    num_agents = 500
    degree = 3

    print(f"Constructing random {degree}-regular network with {num_agents} nodes...")
    network = ng.RandomRegularGenerator(num_agents, degree)()

    # Randomly relabel nodes, so that network structure can not be inferred from labels
    new_labels = np.random.permutation(num_agents)
    mapping = {i: new_labels[i] for i in range(num_agents)}
    network = nx.relabel_nodes(network, mapping)
    # print(network.nodes)
    new_network = nx.Graph()
    new_network.add_nodes_from(sorted(network.nodes(data=True)))
    new_network.add_edges_from(network.edges(data=True))
    network = new_network
    # print(network.nodes)

    params = CNVMParameters(
        num_opinions=num_opinions,
        num_agents=num_agents,
        network=network,
        r_imit=1.01,
        r_noise=0.01,
        prob_imit=np.array([[0, 0.99 / 1.01], [1, 0]]),
    )

    save_params("data/params.pkl", params)


def sample_anchors_and_cnvm():
    params = load_params("data/params.pkl")
    num_samples = 100
    num_anchor_points = 1000
    lag_time = 4

    print("Sampling anchor points...")
    x_anchor = ct.create_anchor_points_local_clusters(
        params.network, params.num_opinions, num_anchor_points, 5
    )
    x_anchor = ct.integrate_anchor_points(
        x_anchor, params, lag_time / 10
    )  # integrate shortly to get rid of instable states
    print("Simulating voter model...")
    x_samples = sample_cnvm(x_anchor, num_samples, lag_time, params)

    np.savez_compressed("data/x_data", x_anchor=x_anchor, x_samples=x_samples)


def approximate_tm():
    params = load_params("data/params.pkl")
    x_samples = np.load("data/x_data.npz")["x_samples"]

    sigma = (params.num_agents / 2) ** 0.5
    d = 10

    trans_manifold = TransitionManifold(sigma, 0.015 ** 0.5, d)
    print("Approximating transition manifold...")
    xi = trans_manifold.fit(x_samples)

    np.save("data/dist_mat", trans_manifold.distance_matrix)
    np.save("data/xi", xi)


def estimate_dimension():
    params = load_params("data/params.pkl")
    dist_mat = np.load("data/dist_mat.npy")
    sigma = (params.num_agents / 2) ** 0.5

    trans_manifold = TransitionManifold(sigma, 1)
    trans_manifold.distance_matrix = dist_mat

    print("Estimating dimension...")
    epsilons = np.logspace(-6, 2, 101)
    s = []
    for epsilon in epsilons:
        s.append(trans_manifold.average_kernel_matrix(epsilon))

    np.savez_compressed("data/avg_sim", s=np.array(s), epsilons=epsilons)


def linear_regression():
    num_coordinates = 1
    xi = np.load("data/xi.npy")
    xi = xi[:, :num_coordinates]
    x = np.load("data/x_data.npz")["x_anchor"]
    params = load_params("data/params.pkl")
    network = params.network

    pen_vals = np.logspace(3, -2, 6)
    alphas, colors = optimize_fused_lasso(x, xi, network, pen_vals)

    np.savez("data/cv_optim.npz", alphas=alphas, xi_fit=colors)

    xi_cv = build_cv_from_alpha(alphas, params.num_opinions)
    with open("data/cv.pkl", "wb") as file:
        pickle.dump(xi_cv, file)
