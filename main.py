#!/usr/bin/python3

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import datetime
import requests_cache

cache = requests_cache.CachedSession(cache_name='cache', 
                                     backend='sqlite',
                                     expire_after=datetime.timedelta(days=3))

start_time = datetime.date(2010, 1, 1)

# indexes
sp500 = pdr.DataReader('SP500', 'fred', start_time, session=cache)
djia = pdr.DataReader('DJIA', 'fred', start_time, session=cache)
w5000 = pdr.DataReader('WILL5000INDFC', 'fred', start_time, session=cache)
nasdaq = pdr.DataReader('NASDAQCOM', 'fred', start_time, session=cache)

# bonds
gs1mo_bond = pdr.DataReader('DGS1MO', 'fred', start_time, session=cache)
gs3mo_bond = pdr.DataReader('DGS3MO', 'fred', start_time, session=cache)
gs6mo_bond = pdr.DataReader('DGS6MO', 'fred', start_time, session=cache)
gs1_bond = pdr.DataReader('DGS1', 'fred', start_time, session=cache)
gs2_bond = pdr.DataReader('DGS2', 'fred', start_time, session=cache)
gs5_bond = pdr.DataReader('DGS5', 'fred', start_time, session=cache)
gs10_bond = pdr.DataReader('DGS10', 'fred', start_time, session=cache)

# indicators
vix = pdr.DataReader('VIXCLS', 'fred', start_time, session=cache)

# stocks
shop = pdr.DataReader('SHOP', 'yahoo', start_time, session=cache).iloc[:,3].to_frame()
goog = pdr.DataReader('GOOG', 'yahoo', start_time, session=cache).iloc[:,3].to_frame()
aapl = pdr.DataReader('AAPL', 'yahoo', start_time, session=cache).iloc[:,3].to_frame()

def beta(x, y, start_time=None, end_time=None, freq=None):
    change = x.join(y)
    if start_time:
        change = change.loc[start_time:]
    if end_time:
        change = change.loc[:end_time]
    if freq:
        change = change.resample(freq).ffill()
    change = change.pct_change().dropna()
    c = np.cov(
            change.iloc[:,0].to_numpy().flatten(),
            change.iloc[:,1].to_numpy().flatten())
    return c[0,1]/c[1,1]
