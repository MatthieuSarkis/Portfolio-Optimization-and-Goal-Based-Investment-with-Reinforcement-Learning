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

stocks_symbols = ['MMM','ABT','ABBV','ACN','ATVI','AYI','ADBE','AMD','AAP','AES','AET',
                  'AMG','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE',
                  'AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP',
                  'AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','APC','ADI','ANDV',
                  'ANSS','ANTM','AON','AOS','APA','AIV','AAPL','AMAT','APTV','ADM','ARNC',
                  'AJG','AIZ','T','ADSK','ADP','AZO','AVB','AVY','BHGE','BLL','BAC','BK',
                  'BAX','BBT','BDX','BRK.B','BBY','BIIB','BLK','HRB','BA','BWA','BXP','BSX',
                  'BHF','BMY','AVGO','BB','CHRW','CA','COG','CDNS','CPB','COF','CAH','CBOE',
                  'KMX','CCL','CAT','CBG','CBS','CELG','CNC','CNP','CTL','CERN','CF','SCHW',
                  'CHTR','CHK','CVX','CMG','CB','CHD','CI','XEC','CINF','CTAS','CSCO','C','CFG',
                  'CTXS','CLX','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','CXO','COP',
                  'ED','STZ','COO','GLW','COST','COTY','CCI','CSRA','CSX','CMI','CVS','DHI',
                  'DHR','DRI','DVA','DE','DAL','XRAY','DVN','DLR','DFS','DISCA','DISCK','DISH',
                  'DG','DLTR','D','DOV','DWDP','DPS','DTE','DRE','DUK','DXC','ETFC','EMN','ETN',
                  'EBAY','ECL','EIX','EW','EA','EMR','ETR','EVHC','EOG','EQT','EFX','EQIX','EQR',
                  'ESS','EL','ES','RE','EXC','EXPE','EXPD','ESRX','EXR','XOM','FFIV','FB','FAST',
                  'FRT','FDX','FIS','FITB','FE','FISV','FLIR','FLS','FLR','FMC','FL','F','FTV',
                  'FBHS','BEN','FCX','GPS','GRMN','IT','GD','GE','GGP','GIS','GM','GPC','GILD',
                  'GPN','GS','GT','GWW','HAL','HBI','HOG','HRS','HIG','HAS','HCA','HCP','HP','HSIC',
                  'HSY','HES','HPE','HLT','HOLX','HD','HON','HRL','HST','HPQ','HUM','HBAN','HII',
                  'IDXX','INFO','ITW','ILMN','IR','INTC','ICE','IBM','INCY','IP','IPG','IFF','INTU',
                  'ISRG','IVZ','IQV','IRM','JEC','JBHT','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY',
                  'KMB','KIM','KMI','KLAC','KSS','KHC','KR','LB','LLL','LH','LRCX','LEG','LEN','LUK',
                  'LLY','LNC','LKQ','LMT','L','LOW','LYB','MTB','MAC','M','MRO','MPC','MAR','MMC','MLM',
                  'MAS','MA','MAT','MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','KORS','MCHP','MU',
                  'MSFT','MAA','MHK','TAP','MDLZ','MON','MNST','MCO','MS','MOS','MSI','MYL','NDAQ',
                  'NOV','NAVI','NTAP','NFLX','NWL','NFX','NEM','NWSA','NWS','NEE','NLSN','NKE','NI',
                  'NBL','JWN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','ORLY','OXY','OMC','OKE',
                  'ORCL','PCAR','PKG','PH','PDCO','PAYX','PYPL','PNR','PBCT','PEP','PKI','PRGO','PFE',
                  'PCG','PM','PSX','PNW','PXD','PNC','RL','PPG','PPL','PX','PCLN','PFG','PG','PGR',
                  'PLD','PRU','PEG','PSA','PHM','PVH','QRVO','PWR','QCOM','DGX','RRC','RJF','RTN','O',
                  'RHT','REG','REGN','RF','RSG','RMD','RHI','ROK','COL','ROP','ROST','RCL','CRM','SBAC',
                  'SCG','SLB','SNI','STX','SEE','SPY','SRE','SHW','SIG','SPG','SWKS','SLG','SNA','SO','LUV',
                  'SPGI','SWK','SBUX','STT','SRCL','SYK','STI','SYMC','SYF','SNPS','SYY','TROW','TPR',
                  'TGT','TEL','FTI','TXN','TXT','TMO','TIF','TWX','TJX','TMK','TSS','TSCO','TDG','TRV',
                  'TRIP','FOXA','FOX','TSN','UDR','ULTA','USB','UAA','UA','UNP','UAL','UNH','UPS','URI',
                  'UTX','UHS','UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','VIAB','V','VNO',
                  'VMC','WMT','WBA','DIS','WM','WAT','WEC','WFC','HCN','WDC','WU','WRK','WY','WHR','WMB',
                  'WLTW','WYN','WYNN','XEL','XRX','XLNX','XL','XYL','YUM','ZBH','ZION','ZTS']

stocks_symbols_temp = ['MMM','ABT']
class DataFetcher():
    
    def __init__(self,
                 stock_symbols: list[str] = stocks_symbols,
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
                 df_directory: str = "data",
                 df_file_name: str = "stocks.csv") -> None:
        
        self.df_directory = df_directory
        self.df_file_name = df_file_name
        self.df_path = os.path.join(df_directory, df_file_name)
        self.df = pd.read_csv(self.df_path) 
        self.close_prices = None      
        
    def collect_close_prices(self) -> None:
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        dates = pd.date_range(self.df['Date'].min(), self.df['Date'].max())
        stocks = self.df['Name'].unique()
        assert self.close_prices is None
        self.close_prices = pd.DataFrame(index=dates)

        for stock in stocks:
            df_temp = self.df[self.df['Name'] == stock]
            df_temp2 = pd.DataFrame(data=df_temp['Close'].to_numpy(), index=df_temp['Date'], columns=[stock])
            self.close_prices = pd.concat([self.close_prices, df_temp2], axis=1)  

        close_prices_path = os.path.join(self.df_directory, "close_prices.csv")
        self.close_prices.to_csv(close_prices_path)  
        
    def handle_missing_values(self) -> None:
        
        assert self.close_prices is not None
        self.close_prices.dropna(axis=0, how='all', inplace=True)
        self.close_prices.fillna(method='ffill', inplace=True)
        self.close_prices.fillna(method='bfill', inplace=True)

class Data():
    
    def __init__(self,
                 stock_symbols: list[str] = stocks_symbols,
                 directory: str = "data",
                 file: str = "stocks.csv",
                 start_date: str = "2010-01-01",
                 end_date: str = "2018-12-31") -> None:
        
        self.fetcher = DataFetcher(stock_symbols=stock_symbols,
                                   start_date=start_date,
                                   end_date=end_date,
                                   directory_path=directory)
        
        self.preprocessor = Preprocessor(df_directory=directory,
                                         df_file_name=file)
        
        self.df = None
        
    def generate_preprocessed_data(self) -> pd.DataFrame:
        
        self.fetcher.fetch_and_merge_data()
        #self.preprocessor.collect_close_prices()
        #self.preprocessor.handle_missing_values()
        self.df = self.preprocessor.close_prices
        
        return self.df
        

if __name__ == '__main__':
    
    data = Data(stocks_symbols=stocks_symbols_temp,
                directory='data',
                file='stocks.csv',
                start_date='2010-01-01',
                end_date='2018-12-31')
    
    df = data.generate_preprocessed_data()

    
    #df = pd.read_csv('data/close_prices.csv', index_col=0, parse_dates=True)
    #df = handle_missing_values(df)

           