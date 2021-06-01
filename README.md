## Notes / To do

* Donnees fondamentales?
* Leverage, lower bound on bank account
* tests unitaires: https://www.youtube.com/watch?v=6tNS--WetLI
* typical buy_rate and sell_rate = 0.1%
* Distributional Soft Actor Critic: https://arxiv.org/pdf/2001.02811.pdf
* Implement new buying strategy: trained??

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
--layer1_size 256 \
--layer1_size 256 \
--n_episodes 500 \
--seed 42 \
--mode train \
--memory_size 1000000 \
--initial_date 2015-01-01 \
--final_date 2020-12-31 \
--gpu_devices 0 1 2 3 \
--buy_rule most_first \
--auto_temperature \
```
## License
[Apache License 2.0](https://github.com/MatthieuSarkis/stock/blob/master/LICENSE)
