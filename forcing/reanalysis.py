'''
Created on Oct 4, 2022

@author: simon
'''
import os
import datetime
import pandas as pd

def read_merra_subset(fn, field='T2MMEAN[0][0]'):
    with open(fn, 'r') as f:
        dstr = f.readline().split('.')[-2]
        d = datetime.datetime.strptime(dstr, '%Y%m%d')
        parts = [p.strip() for p in f.readline().split(', ')]
        assert len(parts) == 2
        assert parts[0] == field
        T = float(parts[1])
    return d, T

def load_forcing_merra_subset(folder, to_Celsius=True):
    ld = os.listdir(folder)
    def _match(fn1, fn2, comps=(0, 1, 3, 4)):
        p1, p2 = fn1.split('.'), fn2.split('.')
        return all([p1[co] == p2[co] for co in comps])
    fn0 = ld[0]
    fns = [fn for fn in ld if _match(fn, fn0)]
    vals = [read_merra_subset(os.path.join(folder, fn)) for fn in fns]
    df = pd.DataFrame(vals, columns=('datetime', 'T'))
    df.sort_values(by='datetime', inplace=True)
    df = df.set_index('datetime')
    if to_Celsius:
        df['T'] = df['T'] - 273.15
    return df
    