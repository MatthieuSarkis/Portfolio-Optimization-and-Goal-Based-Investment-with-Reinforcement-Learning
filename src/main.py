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
import numpy as np
import os
import time
import torch

from src.agents import Agent_ManualTemperature, Agent_AutomaticTemperature
from src.environment import Environment
from src.get_data import load_data
from src.run import Run


def main(args):

    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    df = load_data(args.initial_date, args.final_date, args.mode)
    
    env = Environment(stock_market_history=df,
                      initial_cash_in_bank=args.initial_cash,
                      buy_rate=args.buy_rate,
                      sell_rate=args.sell_rate,
                      limit_n_stocks=args.limit_n_stocks,
                      buy_rule=args.buy_rule)
    
    if args.auto_temperature:
        agent_name = 'auto_temperature'
        figure_file = str(args.n_episodes) + 'episodes_AutoTemperature_' + args.mode + '.png'
        agent = Agent_AutomaticTemperature(lr_Q=args.lr_Q,
                                           lr_pi=args.lr_pi, 
                                           lr_alpha=args.lr_alpha,  
                                           agent_name=agent_name, 
                                           input_shape=env.observation_space.shape, 
                                           tau=args.tau,
                                           env=env, 
                                           size=args.memory_size,
                                           batch_size=args.batch_size, 
                                           layer1_size=args.layer1_size, 
                                           layer2_size=args.layer2_size,
                                           action_space_dimension=env.action_space.shape[0],
                                           alpha=args.alpha,
                                           device=device)
    
    else:
        agent_name = 'manual_temperature'
        figure_file = str(args.n_episodes) + 'episodes_ManualTemperature_{}'.format(args.sac_temperature) + args.mode + '.png'
        agent = Agent_ManualTemperature(lr_pi=args.lr_pi, 
                                        lr_Q=args.lr_Q, 
                                        gamma=args.gamma, 
                                        agent_name=agent_name, 
                                        input_shape=env.observation_space.shape, 
                                        tau=args.tau,
                                        env=env, 
                                        size=args.memory_size,
                                        batch_size=args.batch_size, 
                                        layer1_size=args.layer1_size, 
                                        layer2_size=args.layer2_size,
                                        action_space_dimension=env.action_space.shape[0],
                                        device=device)
       
    run = Run(env=env,
              agent=agent,
              n_episodes=args.n_episodes,
              auto_temperature=args.auto_temperature,
              sac_temperature=args.sac_temperature,
              mode=args.mode)
    
    initial_time = time.time()
    
    run.run(log_directory='logs')
    run.plot(figure_file=figure_file)
    
    final_time = time.time()
    
    print('Total training duration: {:.3f}'.format(final_time-initial_time))

if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--initial_cash', type=float, default=10000)
    parser.add_argument('--buy_rate', type=float, default=0.1)
    parser.add_argument('--sell_rate', type=float, default=0.1)
    parser.add_argument('--sac_temperature', type=float, default=2.0)
    parser.add_argument('--limit_n_stocks', type=int, default=50)
    parser.add_argument('--lr_Q', type=float, default=0.0003)
    parser.add_argument('--lr_pi', type=float, default=0.0003)
    parser.add_argument('--lr_alpha', type=float, default=0.0003)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--layer1_size', type=int, default=256)
    parser.add_argument('--layer2_size', type=int, default=256)
    parser.add_argument('--n_episodes', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--seed', type=int, default='42')
    parser.add_argument('--auto_temperature', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--initial_date', type=str, default='2010-01-01')
    parser.add_argument('--final_date', type=str, default='2020-12-31')
    parser.add_argument('--gpu_devices', type=int, nargs='+', default=None)
    parser.add_argument('--buy_rule', type=str, default='most_first')

    args = parser.parse_args()
    main(args)