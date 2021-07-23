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

import os
import pandas as pd
from pathlib import Path
from typing import List
import yfinance as yf

class DataFetcher():
    
    def __init__(self,
                 stock_symbols: List[str],
                 start_date: str = "2010-01-01",
                 end_date: str = "2020-12-31",
                 directory_path: str = "data",
                 ) -> None:
        
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.directory_path = directory_path
        
    def fetch_and_merge_data(self) -> None:
        
        final_df = None
        
        for stock in self.stock_symbols:
            
            file_path = os.path.join(self.directory_path, "{}.csv".format(stock))
            if not os.path.exists(file_path):
                data = yf.download(stock, start=self.start_date, end=self.end_date)
                if data.size > 0:
                    data.to_csv(file_path)
                    file = open(file_path).readlines()
                    if len(file) < 10:
                        os.system("rm " + file_path)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                stock_name = file_path.split('/')[1].split('.')[0]
                df['Name'] = stock_name

                if final_df is None:
                    final_df = df
                else:
                    final_df = final_df.append(df, ignore_index=True)
                
                os.system("rm " + file_path)
                
        path = os.path.join(self.directory_path, 'stocks.csv')
        final_df.to_csv(path, index=False)
    
class Preprocessor():
    
    def __init__(self,
                 df_directory: str = 'data',
                 file_name: str = 'stock.csv',
                 ) -> None:
            
        self.df_directory = df_directory
        path = os.path.join(df_directory, file_name)
        self.df = pd.read_csv(path) 
        
    def collect_close_prices(self) -> pd.DataFrame:
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        dates = pd.date_range(self.df['Date'].min(), self.df['Date'].max())
        stocks = self.df['Name'].unique()
        close_prices = pd.DataFrame(index=dates)

        for stock in stocks:
            df_temp = self.df[self.df['Name'] == stock]
            df_temp2 = pd.DataFrame(data=df_temp['Close'].to_numpy(), index=df_temp['Date'], columns=[stock])
            close_prices = pd.concat([close_prices, df_temp2], axis=1)  
        self.df = close_prices
        return close_prices
        
    def handle_missing_values(self) -> pd.DataFrame:
        
        self.df.dropna(axis=0, how='all', inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        self.df.to_csv(os.path.join(self.df_directory, 'close.csv'))
        return self.df   
  
    
def load_data(initial_date: str, 
              final_date: str, 
              tickers_subset: str,
              mode: str = 'test') -> pd.DataFrame:
    """Wrapper function designed to download, preprocess and load the data into a dataframe.
    
    Args:
        inital_data (str): starting date of the time series
        final_date (str): final date of the time series
        tickers_subset (str): subset of tickers for assets of interest
        mode (str): in order to load either the train or test data
    
    Returns:
        df (pd.DataFrame): multidimensional time series containing the close price of the relevant assets
    """
    
    with open('portfolios_and_tickers/tickers_S&P500.txt') as f:
        stocks_symbols = f.read().splitlines()
      
    if not os.path.exists('data/'):  
        
        print('\n>>>>> Fetching the data <<<<<')
        
        fetcher = DataFetcher(stock_symbols=stocks_symbols,
                              start_date=initial_date,
                              end_date=final_date,
                              directory_path="data")
        
        fetcher.fetch_and_merge_data()
    
    if not os.path.exists('data/close.csv'):
        
        print('>>>>> Extracting close prices <<<<<')
        
        preprocessor = Preprocessor(df_directory='data',
                                    file_name='stocks.csv')
    
        df = preprocessor.collect_close_prices()
        df = preprocessor.handle_missing_values()
    
    else:
        
        print('\n>>>>> Reading the data <<<<<')
        
        df = pd.read_csv('data/close.csv', index_col=0)
        
        # We select the tickers we want to focus on, reading them form a list of tickers text file.
        with open(tickers_subset) as f:
            stocks_subset = f.read().splitlines()
            stocks_subset = [ticker for ticker in stocks_subset if ticker in df.columns]
            
        df = df[stocks_subset]
    
    time_horizon = df.shape[0]
    
    if mode == 'train':
        df = df.iloc[:3*time_horizon//4, :]
    else:
        df = df.iloc[3*time_horizon//4:, :]
        
    return df
