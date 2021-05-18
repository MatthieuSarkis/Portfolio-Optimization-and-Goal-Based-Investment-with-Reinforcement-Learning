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

import numpy as np
import pandas as pd
import gym

class Environment(gym.Env):

    def __init__(self, 
                 stock_market_history: pd.DataFrame,                
                 initial_cash_in_bank: float,
                 buy_rate: float,
                 sell_rate: float,
                 sac_temperature: float,
                 action_scale: float = 50,
                 ) -> None:
        
        super(Environment, self).__init__()
        
        self.stock_market_history = stock_market_history
        self.time_horizon, self.stock_space_dimension = stock_market_history.shape
        
        self.state_space_dimension = 2 * self.stock_space_dimension + 1
        self.action_space_dimension = self.stock_space_dimension
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space_dimension,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_space_dimension,)) 
        self.action_scale = action_scale
        
        self.initial_cash_in_bank = initial_cash_in_bank
        
        self.buy_rate = buy_rate
        self.sell_rate = sell_rate
        self.sac_temperature = sac_temperature
        
        self.current_step = None
        self.cash_in_bank = None
        self.stock_prices = None
        self.number_of_shares = None

        self.reset()
    
    def reset(self) -> np.array:  
        
        self.current_step = 0
        self.cash_in_bank = self.initial_cash_in_bank
        self.stock_prices = self.stock_market_history.iloc[self.current_step]
        self.number_of_shares = np.zeros(self.stock_space_dimension)
        
        return self._get_observation()
        
    def step(self, 
             actions: np.array) -> tuple[np.array, float, bool, dict]:
        
        done = self.current_step == (self.time_horizon - 1)
        self.current_step += 1

        actions = (actions * self.action_scale).astype(int)
        sorted_indices = np.argsort(actions)
                
        initial_value_portfolio = self._get_portfolio_value()
        
        sell_idx = sorted_indices[ : np.where(actions < 0)[0].shape[0]]
        buy_idx = sorted_indices[::-1][ : np.where(actions > 0)[0].shape[0]]

        for idx in sell_idx:  
            self._sell(idx, actions[idx])
            
        for idx in buy_idx: 
            self._buy(idx, actions[idx])
                   
        new_value_portfolio = self._get_portfolio_value()
        info = {'value_portfolio': new_value_portfolio}
        reward = (new_value_portfolio - initial_value_portfolio) * self.sac_temperature    
        
        if self.current_step < self.time_horizon - 1:
            self.stock_prices = self.stock_market_history.iloc[self.current_step]    

        return self._get_observation(), reward, done, info
        
    def _sell(self, 
              idx: int, 
              action: int) -> None:
    
        if int(self.number_of_shares[idx]) < 1:
            return
         
        n_stocks_to_sell = min(-action, int(self.number_of_shares[idx]))
        money_inflow = n_stocks_to_sell * self.stock_prices[idx] * (1 - self.sell_rate)
        self.cash_in_bank += money_inflow
        self.number_of_shares[idx] -= n_stocks_to_sell
            
    def _buy(self, 
             idx: int, 
             action: int) -> None:
        
        if self.cash_in_bank < 0:
            return
        
        n_stocks_to_buy = min(action, self.cash_in_bank // self.stock_prices[idx])
        money_outflow = n_stocks_to_buy * self.stock_prices[idx] * (1 + self.buy_rate)
        self.cash_in_bank -= money_outflow
        self.number_of_shares[idx] += n_stocks_to_buy   
        
    def _get_observation(self) -> np.array:
        
        observation = np.empty(self.state_space_dimension)
        observation[0] = self.cash_in_bank
        observation[1 : self.stock_space_dimension+1] = self.stock_prices
        observation[self.stock_space_dimension+1 : ] = self.number_of_shares
        
        return observation
    
    def _get_portfolio_value(self) -> float:
        
        portfolio_value = self.cash_in_bank + self.number_of_shares.dot(self.stock_prices)
        return portfolio_value
        
        
if __name__ == '__main__':
    
    df = pd.DataFrame([[1,2],[3,4]])
    
    env = Environment(stock_market_history=df,
                      initial_cash_in_bank=10000,
                      buy_rate=0.0,
                      sell_rate=0.0,
                      sac_temperature=1.0,
                      action_scale=50)
    
    env.step(np.array([1.0, 1.0]))
    #env.step(np.array([1.0, 1.0]))
    #env.step(np.array([1.0, 1.0]))
    #env.step(np.array([1.0, -1.0]))
    #env.step(np.array([1.0, -1.0]))
    obs = env._get_observation()
    value = env._get_portfolio_value()
    
    print(value)