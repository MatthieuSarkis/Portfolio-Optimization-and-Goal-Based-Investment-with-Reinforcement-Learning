# Reinforcement Learning for Portfolio Optimization and Goal Based Investment

This repository contains *__work in progress__*, and even though the code runs perfectly well, lots of features and improvements remain to be implemented.

The present project consists in particular of my implementation of various actor-critic reinforcement learning algorithms in PyTorch, applied to the problem of portfolio optimization, and Goal Based Investment.

We define a trading environment class which inherits from the gym.Env class. In the case where one simply focuses on the multidimensional time series for the stock prices, the state space consists of the bank account balance, the price of each asset, and the number of shares owned for each asset.

The action space consist in a n_stocks dimensional box [-1, 1]^n_stocks. An action in this continuous action space is then rescaled by a fixed factor and cast to an integer to obtain the number of shares to buy or sell for each asset.

An agent then evolves in this environment, and at  each time step (typically 1 day in the case of the S&P500 data fetched from Yahoo Finance).

The spaces under consideration being of high dimension, we focus in a first step on stochastic policy algorithms. In particular we consider three different variants of Soft Actor Critic, corresponding to the following three papers:
* SAC v1: https://arxiv.org/abs/1801.01290
* SAC v3: https://arxiv.org/abs/1812.05905
* Distributional SAC: https://arxiv.org/pdf/2001.02811.pdf

## Requirements

* python>=3.6
* numpy
* pandas
* jupyter
* sklearn
* matplotlib
* seaborn
* pytorch
* yfinance
* gym

```shell
pip install -r requirements.txt
python setup.py install
```
 ## Examples 
 ### __Create or load a model__, train it or test it:

Simply modify the job.sh script and run:
```shell
./job.sh
```

Or manually:
```shell
python src/main.py \
--initial_cash 100000 \
--buy_rate 0.0001 \
--sell_rate 0.0001 \
--bank_rate 0.5 \
--sac_temperature 1.0 \
--limit_n_stocks 100 \
--lr_Q 0.0003 \
--lr_pi 0.0003 \
--lr_alpha 0.0003 \
--gamma 0.99 \
--tau 0.005 \
--batch_size 256 \
--layer_size 256 \
--n_episodes 1000 \
--seed 0 \
--delay 2 \
--mode train \
--memory_size 1000000 \
--initial_date 2015-01-01 \
--final_date 2020-12-31 \
--gpu_devices 0 1 2 3 \
--grad_clip 2.0 \
--buy_rule most_first \
--agent_type automatic_temperature \
--window 20 \
--use_corr_matrix \
```

## License
[Apache License 2.0](https://github.com/MatthieuSarkis/Portfolio-Optimization-and-Goal-Based-Investment-with-Reinforcement-Learning/blob/master/LICENSE)


## To do

* Possibility to start from a non-trivial portfolio
* Donnees fondamentales des entreprises, news
* More generally, better data engineering. Dimensionality reduction along stock space dimension instead of simply plugging in the correlation matrix?
* Leverage, lower bound on bank account
* tests unitaires: https://www.youtube.com/watch?v=6tNS--WetLI
* Implement Prioritized Experience Replay, or even better: https://arxiv.org/abs/1906.04009
* Cosine annealing for learning rates?
* Different types of deep neural nets? 
* How about redefining what we call an observation in the environment? An observation could be a sequence of n time steps for instance. We could use a wrapper for that.
* Improve command line arguments parsing
* Benchmark
* Generate better logs, define some Log class
* Distributed algorithms

## Done

* Two-timescale update: update the policy and temperature every m>1 iterations (cf. https://arxiv.org/pdf/1802.09477.pdf)
* Implement Distributional Soft Actor Critic
* Implement new buying strategy: random, cyclic
* Use GELU instead of RELU activation? (cf. https://arxiv.org/pdf/1606.08415.pdf)
* Save hyperparameters in json file
* Add correlation matrix (defined by a sliding window) to the data. It is just a time dependent matrix
* Possibility to append a certain number of eigenvalues of the correlation matrix
* Commenter le code en details
