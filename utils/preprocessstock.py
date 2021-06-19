import numpy as np
import pandas as pd
import math, os, sys, datetime
from pandas_datareader import data as pdr
from datetime import date, timedelta
import yfinance as yf
yf.pdr_override()
from sklearn import preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from functools import reduce
from ta.utils import dropna
from ta import add_all_ta_features

def preprocess(stock:str, timesteps:int=7):
    
    startdate = datetime.datetime(2017, 1, 13)
    enddate = datetime.datetime(2021, 1, 1)
    enddate2 = datetime.datetime(2021, 2, 19)

    ticker = yf.Ticker(stock)
    data = ticker.history(start=startdate, end=enddate)
    
    # Create Label
    data['next_Close'] = data['Close'].shift(-7)
    data = data.drop(columns=['Dividends', 'Stock Splits'])
    data = dropna(data)
    
    # Add Indicator
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    data = data.drop(columns=['Volume', 'Open', 'High', 'Low'])
    
    # Feature Selection
    y = data['next_Close']
    featureScores = pd.DataFrame(data[data.columns[1:]].corr()['next_Close'][:])
    x_list = []
    for i in range(0, len(featureScores)):
        if abs(featureScores.next_Close[i]) > 0.90:
            x_list.append(featureScores.index[i])
    X = data[x_list]
    X = X.drop(columns=['next_Close'])
    sfs1 = SFS(LinearRegression(), k_features=(1,5), forward=True, floating=False, cv=0)
    sfs1.fit(X, y)
    k_feature_names = list(sfs1.k_feature_names_)
    features = data[k_feature_names]
    
    # Perporcess
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features = features[:len(features)//timesteps*timesteps].reshape((len(features)//timesteps, timesteps, 5))
    
    labels = data[['next_Close']]
    getmax = labels.max()
    getmin = labels.min()
    labels = min_max_scaler.fit_transform(labels)
#     labels = data['next_Close'].to_numpy()
    labels = labels[:len(labels)//timesteps*timesteps].reshape((len(labels)//timesteps, timesteps, 1))
    labels = np.squeeze(labels)
        
    return features, labels, getmax, getmin

def get_volumes(stock):
    startdate = datetime.datetime(2017, 1, 13)
    enddate = datetime.datetime(2021, 1, 1)
    ticker = yf.Ticker(stock)
    data = ticker.history(start=startdate, end=enddate)
    data.reset_index(inplace=True)
    volumes = data['Volume'].to_numpy()

    return volumes
