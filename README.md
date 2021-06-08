## Bugs to solve

* pour DSAC, il semble que ce soit le alpha -> nan
* pour auto temp c'est mu
## Notes

* typical buy_rate and sell_rate = 0.1%
## To do

* Donnees fondamentales des entreprises, news
* Leverage, lower bound on bank account
* tests unitaires: https://www.youtube.com/watch?v=6tNS--WetLI
* Prioritized Experience Replay, or even better: https://arxiv.org/abs/1906.04009
* test and debug DSAC
* Cosine annealing for learning rates?
* Different types of deep neural nets?
* Better data preprocessing? dimensionality reduction along stock space dimension instead of just plugging in the correlation matrix?
* How about redefining what we call an observation in the environment? An observation could be a sequence of n time steps for instance. We could use a wrapper for that.
* Ameliorer les arguments en ligne de commande avec des if statements et des help
## Done

* Two-timescale update: update the policy and temperature every m>1 iterations (cf. https://arxiv.org/pdf/1802.09477.pdf)
* Distributional Soft Actor Critic: https://arxiv.org/pdf/2001.02811.pdf
                                    https://www.researchgate.net/publication/341069321_DSAC_Distributional_Soft_Actor_Critic_for_Risk-Sensitive_Learning
* Implement new buying strategy: random, cyclic
* Use GELU instead of RELU activation? (cf. https://arxiv.org/pdf/1606.08415.pdf)
* Save hyperparameters in json file
* Add correlation matrix (defined by a sliding window) to the data. It is just a time dependent matrix
* Commenter le code en details

## Requirements

* Python>=3.6
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
 ## Example 
 ### __Create a model__, train it, and save the trained model.

```shell
python src/main.py \
--initial_cash 100000 \
--buy_rate 0.01 \
--sell_rate 0.01 \
--bank_rate 0.0 \
--sac_temperature 2.0 \
--limit_n_stocks 500 \
--lr_Q 0.0003 \
--lr_pi 0.0003 \
--gamma 0.99 \
--tau 0.005 \
--batch_size 256 \
--layer_size 256 \
--n_episodes 5 \
--seed 42 \
--mode train \
--memory_size 1000000 \
--initial_date 2015-01-01 \
--final_date 2020-12-31 \
--gpu_devices 0 1 2 3 \
--buy_rule most_first \
--agent_type automatic_temperature \
--window 20 \
--use_corr_matrix \
```
## License
[Apache License 2.0](https://github.com/MatthieuSarkis/stock/blob/master/LICENSE)
