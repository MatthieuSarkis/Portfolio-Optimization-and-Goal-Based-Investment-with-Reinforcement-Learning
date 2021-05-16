import numpy as np
import pandas as pd
import gym

class Environment(gym.Env):

    def __init__(self, 
                 df: pd.DataFrame, 
                 stock_space_dimension: int,                
                 initial_cash_in_bank: float,
                 buy_rate: float,
                 sell_rate: float,
                 sac_temperature: float,
                 state_space_dimension: int,
                 action_space_dimension: int,
                 action_scale: float = 50,
                 ) -> None:
        
        super(Environment, self).__init__()
        
        self.df = df
        self.stock_space_dimension = stock_space_dimension
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_space_dimension,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_space_dimension,)) 
        self.action_scale = action_scale
        self.initial_cash_in_bank = initial_cash_in_bank
        self.buy_rate = buy_rate
        self.sell_rate = sell_rate
        self.sac_temperature = sac_temperature
        
        self.current_step = None
        self.data = None
        self.state = None
        self.reward = None

        self.reset()
    
    def reset(self) -> list[float]:  
        
        self.state = [self.initial_cash_in_bank] + self.data.Close.values.tolist() + [0] * self.stock_space_dimension
        self.current_step = 0
        self.data = self.df.loc[self.current_step,:]
        self.reward = 0
        return self.state
    
    def step(self, 
             actions: np.array) -> tuple[list[float], float, bool, dict]:
        
        done = self.current_step == len(self.df.index.unique()) - 1
        self.current_step += 1
        
        actions = (actions * self.action_scale).astype(int)
        sorted_indices = np.argsort(actions)
        
        initial_value_portfolio = self.state[0] + sum(np.array(self.state[1:(self.stock_space_dimension+1)]) * np.array(self.state[(self.stock_space_dimension+1):(self.stock_space_dimension*2+1)]))
        
        sell_idx = sorted_indices[:np.where(actions < 0)[0].shape[0]]
        buy_idx = sorted_indices[::-1][:np.where(actions > 0)[0].shape[0]]

        for idx in sell_idx:  
            self._sell(idx, actions[idx])
            
        for idx in buy_idx: 
            self._buy(idx, actions[idx])

        self.data = self.df.loc[self.current_step, :]    
        self.state = [self.state[0]] + self.data.Close.values.tolist() + list(self.state[(self.stock_space_dimension+1):(self.stock_space_dimension*2+1)])
                        
        value_portfolio = self.state[0] + sum(np.array(self.state[1:(self.stock_space_dimension+1)]) * np.array(self.state[(self.stock_space_dimension+1):(self.stock_space_dimension*2+1)]))
        info = {'value_portfolio': value_portfolio}
        
        self.reward = (value_portfolio - initial_value_portfolio) * self.sac_temperature         

        return self.state, self.reward, done, info
        
    def _sell(self, 
              idx: int, 
              action: int) -> None:
    
        if self.state[idx+self.stock_space_dimension+1] <= 0:
            return
         
        n_stocks_to_sell = min(-action, self.state[idx+self.stock_space_dimension+1])
        money_inflow = self.state[idx+1] * n_stocks_to_sell * (1 - self.sell_rate)
        self.state[0] += money_inflow
        self.state[idx+self.stock_space_dimension+1] -= n_stocks_to_sell
            
    def _buy(self, 
             idx: int, 
             action: int) -> None:
        
        n_stocks_to_buy = min(action, self.state[0] // self.state[idx+1])
        money_outflow = self.state[idx+1] * n_stocks_to_buy * (1 + self.buy_rate)
        self.state[0] -= money_outflow
        self.state[idx+self.stock_space_dimension+1] += n_stocks_to_buy   
        