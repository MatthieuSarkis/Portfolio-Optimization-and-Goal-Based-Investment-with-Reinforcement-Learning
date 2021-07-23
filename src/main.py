# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from argparse import ArgumentParser
from datetime import datetime
import json 
import numpy as np
import os
import time
import torch

from src.agents import instanciate_agent
from src.environment import Environment
from src.get_data import load_data
from pathlib import Path
from src.run import Run
from src.utilities import instanciate_scaler, prepare_initial_portfolio


def main(args):

    # creating all the necessary directory tree structure for efficient logging
    date: str = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    checkpoint_directory = os.path.join("saved_outputs", date) if args.mode=='train' else args.checkpoint_directory
    checkpoint_directory_networks = os.path.join(checkpoint_directory, "networks")
    checkpoint_directory_logs = os.path.join(checkpoint_directory, "logs")
    checkpoint_directory_plots = os.path.join(checkpoint_directory, "plots")
    Path(checkpoint_directory_networks).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_logs).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_plots).mkdir(parents=True, exist_ok=True)
    # write the checkpoint directory name in a file for quick access when testing
    if args.mode == 'train':
        with open(os.path.join(checkpoint_directory, "checkpoint_directory.txt"), "w") as f:
            f.write(checkpoint_directory) 

    # saving the (hyper)parameters used 
    params_dict = vars(args)
    with open(os.path.join(checkpoint_directory, "parameters.json"), "w") as f: 
        json.dump(params_dict, f, indent=4)
    
    # specifying the hardware
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initializing the random seeds for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
   
    # downloading, preprocessing and loading the multidimensional time series into a dataframe
    df = load_data(initial_date=args.initial_date, 
                   final_date=args.final_date, 
                   tickers_subset=args.assets_to_trade,
                   mode=args.mode)
    
    # preparing the initial portfolio to pass it to the constructor of the environment 
    initial_portfolio = prepare_initial_portfolio(initial_portfolio=args.initial_cash if args.initial_cash is not None else args.initial_portfolio,
                                                  tickers=df.columns.to_list())
     
    # instanciating the trading environment   
    env = Environment(stock_market_history=df,
                      initial_portfolio=initial_portfolio,
                      buy_cost=args.buy_cost,
                      sell_cost=args.sell_cost,
                      bank_rate=args.bank_rate,
                      limit_n_stocks=args.limit_n_stocks,
                      buy_rule=args.buy_rule,
                      use_corr_matrix=args.use_corr_matrix,
                      use_corr_eigenvalues=args.use_corr_eigenvalues,
                      window=args.window,
                      number_of_eigenvalues=args.number_of_eigenvalues)
    
    # instanciating the data standard scaler
    scaler = instanciate_scaler(env=env, 
                                mode=args.mode,
                                checkpoint_directory=checkpoint_directory_networks)
    
    # instanciating the trading agent
    agent = instanciate_agent(env=env, 
                              device=device, 
                              checkpoint_directory_networks=checkpoint_directory_networks,
                              args=args)
    
    # running the whole training or testing process   
    run = Run(env=env,
              agent=agent,
              n_episodes=args.n_episodes,
              agent_type=args.agent_type,
              mode=args.mode,
              sac_temperature=args.sac_temperature,
              scaler=scaler,
              checkpoint_directory_logs=checkpoint_directory_logs)
    
    initial_time = time.time()
    run.run()
    run.plot(checkpoint_directory_plots=checkpoint_directory_plots)
    final_time = time.time()
    print('Total training duration: {:*^13.3f}'.format(final_time-initial_time))


if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--assets_to_trade',       type=str,            default='./portfolios_and_tickers/tickers_S&P500.txt', help='')
    parser.add_argument('--initial_cash',          type=float,          default=None,                                          help='')
    parser.add_argument('--initial_portfolio',     type=str,            default='./initial_portfolio.json',                    help='')
    parser.add_argument('--buy_cost',              type=float,          default=0.001,                                         help='')
    parser.add_argument('--sell_cost',             type=float,          default=0.001,                                         help='')
    parser.add_argument('--bank_rate',             type=float,          default=0.5,                                           help='Annual bank rate')
    parser.add_argument('--initial_date',          type=str,            default='2010-01-01',                                  help='')
    parser.add_argument('--final_date',            type=str,            default='2020-12-31',                                  help='')
    parser.add_argument('--limit_n_stocks',        type=int,            default=100,                                           help='')
    parser.add_argument('--agent_type',            type=str,            default='automatic_temperature',                       help='Choose between: manual_temperature, automatic_temperature or distributional')
    parser.add_argument('--buy_rule',              type=str,            default='most_first',                                  help='')
    parser.add_argument('--sac_temperature',       type=float,          default=2.0,                                           help='')
    parser.add_argument('--gamma',                 type=float,          default=0.99,                                          help='')
    parser.add_argument('--lr_Q',                  type=float,          default=0.0003,                                        help='')
    parser.add_argument('--lr_pi',                 type=float,          default=0.0003,                                        help='')
    parser.add_argument('--lr_alpha',              type=float,          default=0.0003,                                        help='')
    parser.add_argument('--tau',                   type=float,          default=0.005,                                         help='')
    parser.add_argument('--batch_size',            type=int,            default=32,                                            help='')
    parser.add_argument('--layer_size',            type=int,            default=256,                                           help='')
    parser.add_argument('--n_episodes',            type=int,            default=1,                                             help='')
    parser.add_argument('--delay',                 type=int,            default=1,                                             help='')
    parser.add_argument('--memory_size',           type=int,            default=1000000,                                       help='')
    parser.add_argument('--mode',                  type=str,            default='test',                                        help='')
    parser.add_argument('--checkpoint_directory',  type=str,            default=None,                                          help='')
    parser.add_argument('--seed',                  type=int,            default='42',                                          help='')
    parser.add_argument('--gpu_devices',           type=int,            nargs='+', default=[0, 1, 2, 3],                       help='')
    parser.add_argument('--grad_clip',             type=float,          default=1.0,                                           help='')
    parser.add_argument('--window',                type=int,            default=20,                                            help='Window for correlation matrix computation.')
    parser.add_argument('--number_of_eigenvalues', type=int,            default=10,                                            help='Number of largest eigenvalues to append to the close prices time series.')
    parser.add_argument('--use_corr_eigenvalues',  action='store_true', default=False,                                         help='')
    parser.add_argument('--use_corr_matrix',       action='store_true', default=False,                                         help='')
    
    args = parser.parse_args()
    main(args)
