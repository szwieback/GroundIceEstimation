'''
Created on Nov 5, 2021

@author: simon
'''

import os
import pandas as pd
import datetime

def read_toolik_forcing(fn, year=2019):
    assert year == 2019
    def dp(d, t):
        assert isinstance(t, str)
        t_ = str(int(t) - 100)
        dt = d + ' ' + t_.zfill(4)
        dtp = datetime.datetime.strptime(dt, '%Y-%m-%d %H%M')
        dtp = dtp + pd.Timedelta(1, 'h')
        return dtp

    df = pd.read_csv(fn, parse_dates={'datetime': ['date', 'hour']}, date_parser=dp)
    df = df.set_index('datetime')
    return df