'''
Created on Oct 4, 2022

@author: simon
'''
import datetime
import pandas as pd
from forcing.preproc import parse_dates

def read_merra_subset(fn, field='T2MMEAN[0][0]'):
    with open(fn, 'r') as f:
        dstr = f.readline().split('.')[-2]
        d = datetime.datetime.strptime(dstr, '%Y%m%d')
        lines = f.readlines()
        valid = False
        for line in lines:
            parts = [p.strip() for p in line.split(', ')]
            if len(parts) == 2 and parts[0] == field:
                T = float(parts[1])
                valid = True
        if not valid:
            raise ValueError(f"File {fn} could not be parsed")
    return d, T

def load_forcing_merra_subset(folder, to_Celsius=True):
    ld = [f for f in folder.iterdir() if f.is_file() and len(f.suffix) != 4] # ignore .pdf and .txt
    def _match(fn1, fn2, comps=(0, 1, 3, 4)):
        p1, p2 = str(fn1).split('.'), str(fn2).split('.')
        return all([p1[co] == p2[co] for co in comps])
    fn0 = ld[0]  # wonky
    fns = [fn for fn in ld if _match(fn, fn0)]
    vals = [read_merra_subset(folder / fn) for fn in fns]
    df = pd.DataFrame(vals, columns=('datetime', 'T'))
    df.sort_values(by='datetime', inplace=True)
    df = df.set_index('datetime')
    if to_Celsius:
        df['T'] = df['T'] - 273.15
    return df

def forcing_merra_meta(pforcing, meta, year=None, dateformat='%Y%m%d'):
    df = load_forcing_merra_subset(pforcing, to_Celsius=True)
    d0, d1 = parse_dates(meta['date_interval'], strp=dateformat)
    datesdisp = parse_dates(meta['dates_scenes'], strp=dateformat)
    if year is None: year = d0.year
    if d0.year != year or d1.year != year or any([d.year != year for d in datesdisp]):
        raise ValueError("Years must be consistent")
    dailytemp = df.resample('D').mean()['T'].loc[d0:d1]
    dailytemp[dailytemp < 0] = 0
    ind_scenes = [int((d - d0).days) for d in datesdisp]
    return dailytemp, ind_scenes
