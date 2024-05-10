# `supergraph` package

[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Continuous Integration](https://github.com/bheijden/supergraph/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/bheijden/supergraph/actions/workflows/ci.yml)
[![Test Coverage](coverage.svg)](https://github.com/bheijden/supergraph/actions/workflows/ci.yml)


What is the `supergraph` package
-------------------------------------

The supergraph package provides a framework for simulating cyber-physical systems with realistic asynchronicity and delays. It extends conventional accelerated physics simulations by encoding data dependencies across parallel simulation steps into a supergraph, minimizing redundant computation and bridging the sim2real gap. The package includes examples that demonstrate its implementation and evaluation:

1. **Environment Compilation:** Demonstrates compiling computation graphs into robotic environments using [Brax](https://github.com/google/brax) as the physics engine, with support for easily integrating other environments like [Mujoco's MJX](https://mujoco.readthedocs.io/en/stable/mjx.html). This example utilizes the supergraph method to simulate efficient delay capabilities in a gym-like interface. [View on Google Colab](https://colab.research.google.com/github/bheijden/supergraph/blob/master/notebooks/compiler.ipynb)

2. **Supergraph Algorithm:** Detailed implementation of key algorithms with both readable and optimized versions, showcasing the use of supergraphs in simulation. [View on Google Colab](https://colab.research.google.com/github/bheijden/supergraph/blob/master/notebooks/code_example.ipynb)

Installation
------------

You can install the package using pip:

```bash
pip3 install supergraph
```

Cite supergraph
-----------

If you are using supergraph for your scientific publications, please cite:

``` {.sourceCode .bibtex}
@article{heijden2024efficient,
  title={Efficient Parallelized Simulation of Cyber-Physical Systems},
  author={van der Heijden, Bas and Ferranti, Laura and Kober, Jens and Babuska, Robert},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```

Acknowledgements
----------------

This project is funded by the [OpenDR](https://opendr.eu/) Horizon 2020 project.
