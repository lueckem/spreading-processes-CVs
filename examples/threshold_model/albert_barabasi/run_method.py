import pickle
from sponet import CNTMParameters, save_params, load_params
import numpy as np
import networkx as nx
from sponet import network_generator as ng

from interpretable_cvs import (
    TransitionManifold,
    optimize_fused_lasso,
    sample_cntm,
    build_cv_from_alpha,
)
import interpretable_cvs as ct


def setup_params():
    num_opinions = 2
    num_agents = 500

    print(f"Constructing Albert-Barabasi model with {num_agents} nodes...")
    network = ng.BarabasiAlbertGenerator(num_agents, 2)()

    # Randomly relabel nodes, so that network structure can not be inferred from labels
    new_labels = np.random.permutation(num_agents)
    mapping = {i: new_labels[i] for i in range(num_agents)}
    network = nx.relabel_nodes(network, mapping)

    new_network = nx.Graph()
    new_network.add_nodes_from(sorted(network.nodes(data=True)))
    new_network.add_edges_from(network.edges(data=True))
    network = new_network

    params = CNTMParameters(
        network=network,
        r=1,
        r_tilde=0.1,
        threshold_01=0.5,
        threshold_10=0.5,
    )

    save_params("data/params.pkl", params)


def sample_anchors_and_cnvm():
    params = load_params("data/params.pkl")
    num_samples = 100
    num_anchor_points = 1000
    lag_time = 2

    print("Sampling anchor points...")
    x_anchor = ct.create_anchor_points_local_clusters(
        params.network, 2, num_anchor_points, 5
    )

    print("Simulating threshold model...")
    x_samples = sample_cntm(x_anchor, num_samples, lag_time, params)

    np.savez_compressed("data/x_data", x_anchor=x_anchor, x_samples=x_samples)


def approximate_tm():
    params = load_params("data/params.pkl")
    x_samples = np.load("data/x_data.npz")["x_samples"]

    sigma = (params.num_agents / 2) ** 0.5
    d = 3

    trans_manifold = TransitionManifold(sigma, 1, d)
    print("Approximating transition manifold...")
    xi = trans_manifold.fit(x_samples)

    np.save("data/xi", xi)


def linear_regression():
    num_coordinates = 3
    xi = np.load("data/xi.npy")
    xi = xi[:, :num_coordinates]
    x = np.load("data/x_data.npz")["x_anchor"]
    params = load_params("data/params.pkl")
    network = params.network

    # no pre-weighting
    pen_vals = np.logspace(3, -2, 6)
    alphas, colors = optimize_fused_lasso(x, xi, network, pen_vals)

    np.savez("data/cv_optim.npz", alphas=alphas, xi_fit=colors)

    xi_cv = build_cv_from_alpha(alphas, 2)
    with open("data/cv.pkl", "wb") as file:
        pickle.dump(xi_cv, file)

    # pre-weighting
    weights = np.array([d for _, d in network.degree()])

    pen_vals = np.logspace(3, -2, 6)
    alphas, colors = optimize_fused_lasso(
        x, xi, network, pen_vals, weights=weights, performance_threshold=0.95
    )

    np.savez("data/cv_optim_degree_weighted.npz", alphas=alphas, xi_fit=colors)

    xi_cv = build_cv_from_alpha(alphas, 2)
    with open("data/cv_degree_weighted.pkl", "wb") as file:
        pickle.dump(xi_cv, file)
