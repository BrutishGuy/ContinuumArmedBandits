
# ContinuumArmedBandits

ContinuumArmedBandits is a Python package for optimizing actions in a continuous domain using Bayesian optimization, tailored for scenarios like optimizing Google Ad spend in a marketing strategy. The approach generalizes the multi-armed bandit problem to continuous actions, aiming to maximize reward signals across various contexts.

## Table of Contents

- [Introduction](#introduction)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Additional Materials](#additional-materials)
- [Contributing](#contributing)
- [License](#license)

## Introduction

ContinuumArmedBandits leverages Bayesian optimization and Gaussian processes to efficiently explore and exploit a continuous action space. This is particularly useful in applications where sampling the function to be optimized is expensive, such as in marketing strategies to optimize ad spend on critical keywords for Google Search.

## How It Works

Bayesian optimization constructs a posterior distribution of the target function using a Gaussian process. This distribution improves as more observations are collected, guiding the algorithm to explore promising areas while exploiting known good regions. This process is iterative and balances exploration and exploitation using strategies like Upper Confidence Bound (UCB) or Expected Improvement (EI).

![BayesianOptimization in action](https://github.com/BrutishGuy/ContinuumArmedBandits/tree/master/docs/bayesian_optimization)

## Installation

To install the package, clone the repository and install the required dependencies:

```bash
git clone https://github.com/BrutishGuy/ContinuumArmedBandits.git
cd ContinuumArmedBandits
pip install -r requirements.txt
```

## Getting Started - Bayesian Optimization Example

Here’s a quick example to get you started with optimizing a black-box function.

### Define the Function

Define the function you wish to optimize. In a real scenario, the function's internals are unknown.

```python
def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1
```

### Setup Bayesian Optimization

Instantiate the `BayesianOptimization` object with the function and parameter bounds.

```python
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)
```

### Run Optimization

Run the optimizer to find the optimal parameters.

```python
optimizer.maximize(
    init_points=2,
    n_iter=3,
)

print(optimizer.max)
```

## Additional Materials 

### Bayesian Optimization

Explore the detailed examples and advanced usage in the provided notebooks and scripts in the repository:

- [Visualization Notebook](https://github.com/BrutishGuy/ContinuumArmedBandits/blob/master/docs/bayesian_optimization/examples/visualization.ipynb)
- [Exporation vs. Exploitation Notebook](https://github.com/BrutishGuy/ContinuumArmedBandits/blob/master/docs/bayesian_optimization/examples/exploitation_vs_exploration.ipynb)
- [Domain Reduction Notebook](https://github.com/BrutishGuy/ContinuumArmedBandits/blob/master/docs/bayesian_optimization/examples/domain_reduction.ipynb)

### MAB and CMAB Resources

This set of materials was last updated August 2021, so do additional research. There are always new and interesting pieces of work and research coming out which may be of interest to folks.

Below you can find links to various reading materials relating to bandits, the contextual variants, how to evaluate bandit algorithms, variations of bandits for continuous action domains, for ranking problems, etc.

- [Introduction to Various Multi-Armed Bandit Algorithms](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/)

- [Evaluation Metrics for Multi-Armed Bandits](https://jamesrledoux.com/algorithms/offline-bandit-evaluation/)

- [Multi-Variate Web Optimization Using Linear Contextual Bandits](https://medium.com/expedia-group-tech/multi-variate-web-optimisation-using-linear-contextual-bandits-567f563cb59)

- [AutoML for Contextual Bandits by Google - Using oracles as environment simulators](https://research.google/pubs/pub48534/)

### Practical Resources

You can below also find some interesting implementations of some of the theory presented here and generally speaking some useful bandit libraries on GitHub.

- [Bandit Ranking algorithms](https://github.com/tdunning/bandit-ranking)

- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)

- [Gaussian Process Contextual Bandits (Assumes continuous contexts, unfortunately)](https://github.com/ardaegeunlu/Contextual-Gaussian-Process-Bandit-Optimization)

- [Real Time Bidding Agent using RL](https://github.com/venkatacrc/Budget_Constrained_Bidding/blob/master/src/rtb_agent/rl_bid_agent.py)

- [BanditLib - A Bandit library with many implementations of bandit algorithms](https://github.com/huazhengwang/BanditLib)

- [Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertisement](https://github.com/ostigg/dqn-rtb)

- [Contextual Bandits using Decision Trees to Reduce the Search Space](https://github.com/Mathtodon/Contextual_Bandits_Tree)

- [HATCH](https://github.com/ymy4323460/HATCH)

- [Multi-Armed Bandit Algorithms](https://github.com/alison-carrera/mabalgs)

- [More Contextual Bandit algorithm implementations by David Cortes](https://github.com/david-cortes/contextualbandits)

- [Real-Time Bidding by Reinforcement Learning in Display Advertising](https://github.com/han-cai/rlb-dp)

## Contributing

Please open issues for bugs or feature requests, and submit pull requests for review.

## License

This project is licensed under the MIT License.
