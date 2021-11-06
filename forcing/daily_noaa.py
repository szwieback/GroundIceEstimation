'''
Created on Nov 4, 2021

@author: simon
'''
import pandas as pd

def read_daily_noaa_forcing(fn, field='TAVG', convert_temperature=True):
    df = pd.read_csv(fn, parse_dates={'datetime':['DATE']})
    df = df.set_index('datetime')
    df = df[field]
    if convert_temperature:
        df = (df - 32) / 1.8
    return df
