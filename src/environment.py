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

import gym
import numpy as np
import pandas as pd
from typing import Tuple

from src.utilities import append_corr_matrix, append_corr_matrix_eigenvalues
class Environment(gym.Env):
    """Environment for stock trading.
    
    Inherits from gym.Env, to access gym.spaces.Box for the state space and action space.
    
    Attributes:
        observation_space (gym.spaces.Box): (bank account balance, stocks price, corr matrix, owned shares)
        action_space (gym.spaces.Box): cube [-1,1]^n_stocks. positive value: buy, negative value: sale
    """

    def __init__(self, 
                 stock_market_history: pd.DataFrame,                
                 initial_portfolio: dict,
                 buy_cost: float = 0.001,
                 sell_cost: float = 0.001,
                 bank_rate: float = 0.5,
                 limit_n_stocks: float = 200,
                 buy_rule: str = 'most_first',
                 use_corr_matrix: bool = False,
                 use_corr_eigenvalues: bool = False,
                 window: int = 20,
                 number_of_eigenvalues: int = 10,
                 ) -> None:
        """Initialize the Environment object.

        Args:
            stock_market_history (pd.DataFrame): (time_horizon * number_stocks) format                
            initial_portfolio (dict): Initial structure of the portfolio (cash in bank and shares owned)
            buy_cost (float): fees in percentage for buying a stock 
            sell_cost (float): fees in percentage for selling a stock 
            bank_rate (float): annual interest rate of the bank account
            limit_n_stocks (float): maximum number of stocks one can buy or sell at once
            buy_rule (str): specifies the order in which one buys the stocks the agent decided to buy
            use_corr_matrix (bool): whether or not to append the correlation matrix to the time series
            window (int): in case the correlation matrix is used, size of the sliding window

        Returns:
            no value
        """
        
        super(Environment, self).__init__()
        
        self.stock_market_history = stock_market_history
        self.stock_space_dimension = stock_market_history.shape[1]
        self.buy_rule = buy_rule
        self.use_corr_matrix = use_corr_matrix
        self.use_corr_eigenvalues = use_corr_eigenvalues
        
        if self.use_corr_matrix:
            self.stock_market_history = append_corr_matrix(df=self.stock_market_history,
                                                           window=window)
            
        elif self.use_corr_eigenvalues:
            self.stock_market_history = append_corr_matrix_eigenvalues(df=self.stock_market_history,
                                                                       window=window,
                                                                       number_of_eigenvalues = number_of_eigenvalues)
        
        self.time_horizon = self.stock_market_history.shape[0]
        
        self.state_space_dimension = 1 + self.stock_space_dimension + self.stock_market_history.shape[1]
        self.action_space_dimension = self.stock_space_dimension
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space_dimension,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_space_dimension,)) 
        self.limit_n_stocks = limit_n_stocks
        
        self.initial_portfolio = initial_portfolio
        
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.daily_bank_rate = pow(1 + bank_rate, 1 / 365) - 1
        
        self.current_step = None
        self.cash_in_bank = None
        self.stock_prices = None
        self.number_of_shares = None

        self.reset()
    
    def reset(self) -> np.array:  
        """initialize the bank account balance and stock market status 
        the initial time.
        
        Returns:
            np.array of the initial observation 
        """
        
        self.current_step = 0
        self.cash_in_bank = self.initial_portfolio["Bank_account"]
        self.stock_prices = self.stock_market_history.iloc[self.current_step]
        self.number_of_shares = np.array([self.initial_portfolio[ticker] for ticker in self.stock_market_history.columns[:self.action_space_dimension]])
        
        return self._get_observation()
        
    def step(self, 
             actions: np.ndarray,
             ) -> Tuple[np.array, float, bool, dict]:
        """Take one step in the trading environment.
        
        Args:
            actions (np.array): continuous action chosen buy the agent
            
        Returns:
            np.array for the new state
        """
         
        self.current_step += 1      
        initial_value_portfolio = self._get_portfolio_value()
        self.stock_prices = self.stock_market_history.iloc[self.current_step] 
        
        self._trade(actions)
        
        self.cash_in_bank *= 1 + self.daily_bank_rate # should this line be before the trade?       
        new_value_portfolio = self._get_portfolio_value()
        done = self.current_step == (self.time_horizon - 1)
        info = {'value_portfolio': new_value_portfolio}
        
        reward = new_value_portfolio - initial_value_portfolio 
           
        return self._get_observation(), reward, done, info
       
    def _trade(self, 
               actions: np.ndarray,
               ) -> None:
        """Perform one trade according to the actions decided buy an agent
        
        Various buying rules are implemented:
        -most_first: buy the stocks with which most shares are to be bought first 
                    since the agent thinks they show an interesting trend
        -cyclic
        -random
        First sell, then buy.
        
        Args:
            actions (np.array): continuous action chosen buy the agent
            
        Returns:
            no value
        """
        
        actions = (actions * self.limit_n_stocks).astype(int)
        sorted_indices = np.argsort(actions)
        
        sell_idx = sorted_indices[ : np.where(actions<0)[0].size]
        buy_idx = sorted_indices[::-1][ : np.where(actions>0)[0].size]

        for idx in sell_idx:  
            self._sell(idx, actions[idx])
        
        if self.buy_rule == 'most_first':
            for idx in buy_idx: 
                self._buy(idx, actions[idx])
                
        if self.buy_rule == 'cyclic':
            should_buy = np.copy(actions[buy_idx])
            i = 0
            while self.cash_in_bank > 0 and not np.all((should_buy == 0)):
                if should_buy[i] > 0:
                    self._buy(buy_idx[i])
                    should_buy[i] -= 1
                i = (i + 1) % len(buy_idx) 
                
        if self.buy_rule == 'random':
            should_buy = np.copy(actions[buy_idx])
            while self.cash_in_bank > 0 and not np.all((should_buy == 0)):
                i = np.random.choice(np.where(should_buy > 0))
                self._buy(buy_idx[i])
                should_buy[i] -= 1
        
    def _sell(self, 
              idx: int, 
              action: int,
              ) -> None:
        """Sell the required amount of stocks, if anything to be sold.
        
        Args:
            idx (int): index of the stock to be sold
            action (int): number of shares of the stock to be sold
            
        Returns:
            no value
        """
    
        if int(self.number_of_shares[idx]) < 1:
            return
         
        n_stocks_to_sell = min(-action, int(self.number_of_shares[idx]))
        money_inflow = n_stocks_to_sell * self.stock_prices[idx] * (1 - self.sell_cost)
        self.cash_in_bank += money_inflow
        self.number_of_shares[idx] -= n_stocks_to_sell
            
    def _buy(self, 
             idx: int, 
             action: int = 1,
             ) -> None:
        """Buy the required amount of stocks, if enough money in the bank account.
        
        Args:
            idx (int): index of the stock to be bought
            action (int): number of shares of the stock to be bought, defaulted at 1 for convenience
            
        Returns:
            no value
        """
        
        if self.cash_in_bank < 0:
            return
        
        n_stocks_to_buy = min(action, self.cash_in_bank // self.stock_prices[idx])
        money_outflow = n_stocks_to_buy * self.stock_prices[idx] * (1 + self.buy_cost)
        self.cash_in_bank -= money_outflow
        self.number_of_shares[idx] += n_stocks_to_buy   
        
    def _get_observation(self) -> np.array:
        """Observation in the format given by the state_space, and perceived by the agent.
        
        Returns:
            np.array for the observation
        """
        
        observation = np.empty(self.state_space_dimension)
        observation[0] = self.cash_in_bank
        observation[1 : self.stock_prices.shape[0]+1] = self.stock_prices
        observation[self.stock_prices.shape[0]+1 : ] = self.number_of_shares
        
        return observation
    
    def _get_portfolio_value(self) -> float:
        """Performs the scalar product of the owned shares and the stock prices and add the bank account.
        
        Returns:
            np.array for the total portfolio value
        """
        
        portfolio_value = self.cash_in_bank + self.number_of_shares.dot(self.stock_prices[:self.stock_space_dimension])
        return portfolio_value