'''
Created on Nov 4, 2021

@author: simon
'''
import pandas as pd
from forcing.preproc import parse_dates

def read_daily_noaa_forcing(fn, field='TAVG', convert_temperature=True):
    df = pd.read_csv(fn, parse_dates={'datetime':['DATE']})
    df = df.set_index('datetime')
    df = df[[field]].rename(columns={field: 'T'})
    if convert_temperature:
        df = (df - 32) / 1.8
    return df

def forcing_daily_noaa_meta(pforcing, meta, year=None, dateformat='%Y%m%d', convert_temperature=False):
    df = read_daily_noaa_forcing(pforcing, convert_temperature=convert_temperature)
    d0, d1 = parse_dates(meta['date_interval'], strp=dateformat)
    datesdisp = parse_dates(meta['dates_scenes'], strp=dateformat)
    if year is None: year = d0.year
    if d0.year != year or d1.year != year or any([d.year != year for d in datesdisp]):
        raise ValueError("Years must be consistent")
    dailytemp = df.resample('D').mean()['T'].loc[d0:d1]
    dailytemp[dailytemp < 0] = 0
    ind_scenes = [int((d - d0).days) for d in datesdisp]
    return dailytemp, ind_scenes