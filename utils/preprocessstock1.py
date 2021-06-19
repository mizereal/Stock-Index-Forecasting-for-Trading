import numpy as np
import pandas as pd
import math, os, sys, datetime
from datetime import date, timedelta
import yfinance as yf
yf.pdr_override()
from functools import reduce
from ta.utils import dropna
from ta.others import DailyLogReturnIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import PercentagePriceOscillator

def preprocess1(filepath:str, save_to:str=None, standardlize_factors:list=None, timesteps:int=7):
    filename = os.path.basename(filepath)
    if filename[filename.rfind('.'):] != '.csv':
        raise Exception('Support only csv file.')
    if not os.path.exists(filepath):
        raise Exception('File not found.')
    if save_to != None:
        if not os.path.exists(save_to):
            raise Exception('can not find save directory')
    timesteps = timesteps if timesteps >= 2 else 2
    df = pd.read_csv(filepath)
    df.insert(0, 'Labels', df['Close'].values)
    data = df
    data['pct_change_tmr'] = 100*((data['Close'].shift(-1) - data['Close'])/data['Close'].shift(-1))
    
    # ADD INDICATORS
    data['pct_change_ytd'] = 100*(data['Close'].pct_change())
    # data['pct_change_ytd'] = 100*((data['Close'] - (data['Close'].shift(1)))/data['Close'].shift(1))
    indicator_obv = OnBalanceVolumeIndicator(close=data["Close"], volume=data['Volume'], fillna=False)
    data['obv'] = indicator_obv.on_balance_volume()
    indicator_ppo_signal = PercentagePriceOscillator(close=data["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    data['ppo_signal'] = indicator_ppo_signal.ppo_signal()
    data = dropna(data)
    
    # Standardlization
    if type( standardlize_factors ) != type( None ):
        if len(standardlize_factors) == 8:
            ytd_mean, ytd_std, obv_mean, obv_std, ppo_signal_mean, ppo_signal_std, pct_change_mean, pct_change_std = standardlize_factors
        else:
            raise Exception('Wrong standardlize_factors.')
    else:
        ytd_mean, ytd_std = data['pct_change_ytd'].mean(), data['pct_change_ytd'].std()
        obv_mean, obv_std = data['obv'].mean(), data['obv'].std()
        ppo_signal_mean, ppo_signal_std = data['ppo_signal'].mean(), data['ppo_signal'].std()
        pct_change_mean, pct_change_std = data['pct_change_tmr'].mean(), data['pct_change_tmr'].std()
        standardlize_factors = ytd_mean, ytd_std, obv_mean, obv_std, ppo_signal_mean, ppo_signal_std, pct_change_mean, pct_change_std
    data['pct_change_ytd'] = (data['pct_change_ytd']-ytd_mean)/ytd_std
    data['obv'] = (data['obv']-obv_mean)/obv_std
    data['ppo_signal'] = (data['ppo_signal']-ppo_signal_mean)/ppo_signal_std
    data['pct_change_tmr'] = (data['pct_change_tmr']-pct_change_mean)/pct_change_std
    stocks  = data.drop(columns=['Open', 'Close', 'Adj Close', 'Volume', 'High', 'Low'])
    if 'Ticker' in df:
        # multiple stocks
        stocks = stocks.groupby(['Ticker'])
        stock_name = []
        stocks_df_list = []
        for name, group in stocks:
            stock_name.append(name)
            stock = group.drop(columns=['Ticker']).to_numpy()
            stocks_df_list.append(stock[:len(stock)//timesteps*timesteps].reshape((len(stock)//timesteps, timesteps, 4))) # trim samples to make it able to devide by timesteps timesteps
        reduce_stocks = reduce(lambda a, b: np.concatenate((a, b)), stocks_df_list) # concate all stock into single array
        reduce_stocks = reduce_stocks if len(reduce_stocks)%2 == 0 else reduce_stocks[:-1] # trim list to make its length equal to even number
        all_data = reduce_stocks
        features, labels = reduce_stocks[0::2][:,:,1:4], reduce_stocks[0::2][:,:,0] # only select ZScoreMidPrice as a feature
        features = np.expand_dims(features, axis=1) # bc it have only 1 feature the shape will be change, need to expand dim to keep the shape as the same
    else:
        # single stock
        stock = stock.to_numpy()
        stock = stock[:len(stock)//timesteps*timesteps].reshape((len(stock)//timesteps, timesteps, 4))
        stock = stock if len(stock)%2 == 0 else stock[:-1]
        all_data = stock
        features, labels = stock[0::2][:,:,1:4], stock[0::2][:,:,0]
        features = np.expand_dims(features, axis=1) # bc it have only 1 feature the shape will be change, need to expand dim to keep the shape as the same
    if save_to != None:
        save_path = os.path.join(save_to, filename[:filename.rfind('.')]+f'-{timesteps}ts-preprocessed.npz')
        with open(save_path, 'wb') as f:
            np.savez(f, all_data=all_data, features=features, labels=labels, standardlize_factors=standardlize_factors)
        return save_path, all_data, features, labels, standardlize_factors
    else:
        return all_data, features, labels, standardlize_factors

def get_volumes(filepath:str):
    filename = os.path.basename(filepath)
    if filename[filename.rfind('.'):] != '.csv':
        raise Exception('Support only csv file.')
    if not os.path.exists(filepath):
        raise Exception('File not found.')
    volumes = None
    try:
        volumes = pd.read_csv(filepath)['Volume'].values
    except Exception as e:
        print(f'Error while reading csv file {e}')
    return volumes