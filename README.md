# Learning Interpretable Collective Variables for Spreading Processes on Networks

This repository contains the code and numerical examples presented in the paper "[Learning interpretable collective variables for spreading processes on networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.109.L022301)" (arXiv: [2307.03491](https://arxiv.org/abs/2307.03491)).

To run the examples, first download the repository and install the needed Python packages via the command
```
pip install .
```
Alternatively, you can set up a virtual environment using `Poetry`.
After installing `Poetry`, run the command
```
poetry install
```
in the package directory.

The scripts to run the examples can be found in the `examples` directory.
For instance, run `examples/voter_model/stochastic_block_model/main.py` to reproduce Example 1 of the main text.
