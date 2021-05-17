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

from pandas.core.frame import DataFrame
import yfinance as yf
import os
import pandas as pd
from glob import glob
from utilities import make_dir

class DataFetcher():
    
    def __init__(self,
                 stock_symbols: list[str],
                 start_date: str = "2010-01-01",
                 end_date: str = "2018-12-31",
                 directory_path: str = "data") -> None:
        
        make_dir(base_dir='.', directory_name=directory_path)
        
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.directory_path = directory_path
        
    """
    def fetch_data(self) -> None:
        
        for stock in self.stock_symbols:
            file_path = os.path.join(self.directory_path, "{}.csv".format(stock))
            if not os.path.exists(file_path):
                data = yf.download(stock, start=self.start_date, end=self.end_date)
                if data.size > 0:
                    data.to_csv(file_path)
                    file = open(file_path).readlines()
                    if len(file) < 10:
                        os.system("rm " + file_path)
                  
    def merge_data(self) -> None:    
        
        files_path = os.path.join(self.directory_path, '*.csv')
        files = glob(files_path)
        
        final_df = None
        
        for file in files:
            df = pd.read_csv(file)
            stock_name = file.split('/')[1].split('.')[0]
            df['Name'] = stock_name

            if final_df is None:
                final_df = df
            else:
                final_df = final_df.append(df, ignore_index=True)

        path = os.path.join(self.directory_path, 'stocks.csv')
        final_df.to_csv(path, index=False)
    """
    
    def fetch_and_merge_data(self) -> pd.DataFrame:
        
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
        
        return final_df
    

class Preprocessor():
    
    def __init__(self,
                 df: pd.DataFrame,
                 df_directory: str = 'data') -> None:
        
        self.df = df    
        self.df_directory = df_directory  
        
    def collect_close_prices(self) -> pd.DataFrame:
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        dates = pd.date_range(self.df['Date'].min(), self.df['Date'].max())
        stocks = self.df['Name'].unique()
        close_prices = pd.DataFrame(index=dates)

        for stock in stocks:
            df_temp = self.df[self.df['Name'] == stock]
            df_temp2 = pd.DataFrame(data=df_temp['Close'].to_numpy(), index=df_temp['Date'], columns=[stock])
            close_prices = pd.concat([close_prices, df_temp2], axis=1)  

        #close_prices_path = os.path.join(self.df_directory, "close_prices.csv")
        #close_prices.to_csv(close_prices_path)
        
        self.df = close_prices
        return close_prices
        
    def handle_missing_values(self) -> pd.DataFrame:
        
        self.df.dropna(axis=0, how='all', inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        
        return self.df

if __name__ == '__main__':
    
    fetcher = DataFetcher(stock_symbols=stocks_symbols_temp,
                          start_date="2010-01-01",
                          end_date="2018-12-31",
                          directory_path="data")
    
    df = fetcher.fetch_and_merge_data()
    
    preprocessor = Preprocessor(df=df,
                                df_directory='data')
    
    df = preprocessor.collect_close_prices()
    df = preprocessor.handle_missing_values()
            
    print(preprocessor.df.head(10))
    

           