from run_method import *
from plot import *
import time


def main():
    run_method()  # run the method, takes ~15 minutes on 16-core CPU
    plot()  # plot results


def run_method():
    start = time.time()

    """ set up the network and parameters of the CNVM """
    setup_params()  # this creates the data file params.pkl

    """ sample the anchor points and run voter model simulations starting at the anchor points """
    sample_anchors_and_cnvm()  # this creates the data file x_data.npz

    """ approximate the transition manifold with a kernel-based algorithm """
    approximate_tm()  # this creates the data file xi.npz

    """ apply linear regression """
    linear_regression()  # this creates the data file cv_optim.npz

    end = time.time()
    print(f"Took {end - start} seconds.")


def plot():
    plot_tm()


if __name__ == "__main__":
    main()
