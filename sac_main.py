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

from environment import Environment
import numpy as np
from sac_agent import Agent
from utilities import make_dir, plot_learning_curve
from get_data import DataFetcher, Preprocessor



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







def main():
    
    # import data here
    
    fetcher = DataFetcher(stock_symbols=stocks_symbols_temp,
                          start_date="2010-01-01",
                          end_date="2018-12-31",
                          directory_path="data")
    
    df = fetcher.fetch_and_merge_data()
    
    preprocessor = Preprocessor(df=df,
                                df_directory='data')
    
    df = preprocessor.collect_close_prices()
    df = preprocessor.handle_missing_values()
    
    # end import data
    
    env_name = 'stock_trading'
    env = Environment(stock_market_history=df,
                      initial_cash_in_bank=10000,
                      buy_rate=0.1,
                      sell_rate=0.1,
                      sac_temperature=2,
                      action_scale=50)
    
    agent = Agent(eta2=0.0003, 
                  eta1=0.0003, 
                  temperature=2, 
                  env_name=env_name, 
                  input_shape=env.observation_space.shape, 
                  tau=0.005,
                  env=env, 
                  batch_size=256, 
                  layer1_size=256, 
                  layer2_size=156,
                  action_space_dimension=env.action_space.shape[0])
    
    n_episodes = 5
    filename = str(n_episodes) + 'episodes_temperature' + str(agent.temperature) + '.png'
    make_dir('plots')
    figure_file = 'plots/' + filename

    best_reward = float('-Inf')
    reward_history = []
    
    load_network_weights = False

    if load_network_weights:
        agent.load_networks()
        env.render(mode='human')
        
    steps = 0
    
    for i in range(n_episodes):
        
        reward = 0
        done = False
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ =  env.step(action)
            steps += 1
            reward += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_network_weights:
                agent.learn()
            observation = observation_
        reward_history.append(reward)
        avg_reward = np.mean(reward_history[-100:])
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            if not load_network_weights:
                agent.save_networks()
        print('episode ', i, 'reward %.1f' % reward, 'trailing 100 episodes average %.1f' % avg_reward,
              'step %d' % steps, env_name, 'temperature', agent.temperature)
    if not load_network_weights:
        x = [i+1 for i in range(n_episodes)]
        plot_learning_curve(x, reward_history, figure_file)



if __name__ == '__main__':
    
    main()
    
    