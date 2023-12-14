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
    num_agents = 1000

    p_in = 0.1
    p_outs = [0.001, 0.003, 0.005, 0.007]
    p_matrices = [np.array([[p_in, p_out], [p_out, p_in]]) for p_out in p_outs]

    print(f"Constructing stochastic block models with {num_agents} nodes...")
    networks = [ng.StochasticBlockGenerator(num_agents, p_matrix)() for p_matrix in p_matrices]

    params = [CNVMParameters(
        num_opinions=num_opinions,
        num_agents=num_agents,
        network=network,
        r_imit=1.01,
        r_noise=0.01,
        prob_imit=np.array([[0, 0.99 / 1.01], [1, 0]]),
    ) for network in networks]

    for i, param in enumerate(params):
        save_params(f"data/params_{i}.pkl", param)


def sample_anchors_and_cnvm():
    num_params = 4
    params = [load_params(f"data/params_{i}.pkl") for i in range(num_params)]
    num_samples = 100
    num_anchor_points = 1000
    lag_time = 2

    print("Sampling anchor points and simulating voter model...")
    for i, param in enumerate(params):
        x_anchor = ct.create_anchor_points_local_clusters(
            param.network, param.num_opinions, num_anchor_points, 3
        )
        x_anchor = ct.integrate_anchor_points(
            x_anchor, param, lag_time / 10
        )  # integrate shortly to get rid of instable states

        x_samples = sample_cnvm(x_anchor, num_samples, lag_time, param)

        np.savez_compressed(f"data/x_data_{i}", x_anchor=x_anchor, x_samples=x_samples)


def approximate_tm():
    num_params = 4
    params = [load_params(f"data/params_{i}.pkl") for i in range(num_params)]
    x_samples = [np.load(f"data/x_data_{i}.npz")["x_samples"] for i in range(num_params)]

    sigma = (params[0].num_agents / 2) ** 0.5
    d = 4

    print("Approximating transition manifold...")
    trans_manifold = TransitionManifold(sigma, 1, d)

    for i, x_sample in enumerate(x_samples):
        xi = trans_manifold.fit(x_sample)
        np.save(f"data/xi_{i}", xi)
