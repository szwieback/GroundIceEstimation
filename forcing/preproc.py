'''
Created on Nov 4, 2021

@author: simon
'''
import datetime
import pandas as pd

def parse_dates(datestr, strp='%Y%m%d'):
    if isinstance(datestr, str):
        return datetime.datetime.strptime(datestr, strp)
    else:
        return [parse_dates(ds, strp=strp) for ds in datestr]
    
def preprocess(
        df, d0, d1, dates, screen_negative=True, resample=True, field=None, strp='%Y%m%d'):
    d0_, d1_ = parse_dates((d0, d1), strp=strp)
    if isinstance(dates[0], str):
        dates = parse_dates(dates, strp=strp)
    if resample:
        df = df.resample('D').mean()
    if field is not None:
        df = df[field]
    dailytemp = df[pd.date_range(start=d0, end=d1)]
    if screen_negative:
        dailytemp[dailytemp < 0] = 0
    ind_scenes = [int((d - d0_).days) for d in dates]
    return dailytemp, ind_scenes