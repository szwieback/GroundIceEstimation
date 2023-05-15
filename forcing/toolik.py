'''
Created on Nov 5, 2021

@author: simon
'''

import os
import pandas as pd
import datetime

def read_toolik_forcing(fn, year=2019):
    assert year == 2019
    df = pd.read_csv(
        fn, parse_dates={'datetime': ['date', 'hour']})
    def parse(s):
        _dt = datetime.datetime(int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:-2])-1)
        return _dt + pd.Timedelta(1, 'h')
    
    dt = pd.to_datetime([parse(d) for d in df['datetime']], format='%Y-%m-%d %H%M')
    df['datetime'] = dt
    df = df.set_index('datetime')
    return df

from scripts.pathnames import paths
from forcing import parse_dates
fnforcing = os.path.join(paths['forcing'], 'toolik2019', '1-hour_data.csv')
df = read_toolik_forcing(fnforcing)
d0, d1 = '2019-05-28', '2019-09-15'
d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
