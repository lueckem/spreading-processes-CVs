import pickle

import matplotlib.pyplot as plt
from sponet import Parameters, save_params, load_params, sample_many_runs
from sponet.collective_variables import OpinionShares
import numpy as np
import networkx as nx


def main():
    num_samples = 100
    params = load_params("data/params.pkl")
    x_init = np.random.randint(0, 2, size=(1, params.num_agents))
    cv = OpinionShares(2, True, idx_to_return=0)
    x_anchor = np.load("data/x_data.npz")["x_anchor"]

    t, x = sample_many_runs(params, x_anchor[900:905, :], 30, 500, num_samples, -1, collective_variable=cv)

    fig, ax = plt.subplots(nrows=4)
    for j in range(4):
        for i in range(num_samples):
            ax[j].plot(t, x[j, i, :, 0])
    plt.show()


if __name__ == '__main__':
    main()
