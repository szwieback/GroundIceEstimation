'''
Created on Feb 17, 2025

@author: simon
'''

import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from collections import namedtuple

from forcing import load_forcing_merra_subset, parse_dates
from analysis import (StefanPredictor, PredictionEnsemble, InversionResultsISMmap)
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple)
from scripts.pathnames import paths
from scripts.plot_profile import read_InSAR

wavelength = 0.055
geom = {'ia': 37.70 / 180 * np.pi}

Scenario = namedtuple('Scenario', ['name', 'year', 'remove_last', 'reference', 'extended_metrics'])

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.4, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.20, 'low_horizon': 0.10, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}

ll, ur = None, None

datesstr = {2023:
                 ('20230516', '20230528', '20230609', '20230621', '20230703', '20230715', '20230727', 
                  '20230808', '20230901', '20230913', '20230925')}
# xy_ref = np.array([-166.4759, 65.3511])[:, np.newaxis]
xy_ref = np.array([-139.88096, 67.61312])[:, np.newaxis]

var_atmo = (4e-3) ** 2
fns_unw_offset = {}


def oldcrow_forcing(folder_forcing, year, remove_last=True):
    df = load_forcing_merra_subset(folder_forcing)
    d0 = {2023: '2023-05-20'}[year]
    d1 = {2023: '2023-09-26'}[year]
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    if remove_last: ind_scenes = ind_scenes[:-1]
    dailytemp = (df.resample('D').mean())['T'].loc[pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    return dailytemp, ind_scenes


def process_oldcrow(year=2023, rmethod='hadamard', remove_last=True):
    from analysis import (InversionProcessorIS, InversionResultsIS)    
    path0 = paths['stacks'] / f'OldCrow/{year}/proc/{rmethod}/geocoded'
    folder_forcing = paths['forcing'] / 'oldcrow'
    pathout = paths['processed'] / f'oldcrow/{year}/{rmethod}'
    N = 10000
    Nbatch = 1

    s_obs, K, geospatial = read_InSAR(
        path0, wavelength, xy_ref=xy_ref, fns_unw_offset=fns_unw_offset.get(year, []), var_atmo=var_atmo,
        fill_nan=True)
    dailytemp, ind_scenes = oldcrow_forcing(folder_forcing, year=year, remove_last=remove_last)
    indranges = [(ind_scenes[-4], ind_scenes[-1])] 


    predictor = StefanPredictor()
    
    Strat = StefanStratigraphySmoothingSpline
    strat = StratigraphyMultiple(Strat(N=N, dist=params_distribution), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    predens.predict_mean_period(indranges)
    if remove_last:
        s_obs = s_obs[:-1, ...]
        K = K[:-1,:-1, ...]
    data = {'s_obs': s_obs, 'K': K}
    for dname in data.keys():
        data[dname], geospatial_crop = geospatial.crop(data[dname], ll=ll, ur=ur)
        
    ip = InversionProcessorIS(predens, geospatial=geospatial_crop)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], pathout=pathout, n_jobs=-1, overwrite=True, memory=False)
    ir.save(pathout / 'ir.p')
    ip.delete_temporary(pathout)
    ir = InversionResultsISMmap.from_file(pathout / 'ir.p')
    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
        ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e_mean_period', 'var'), ('e_mean_period', 'mean'),
        ('e_mean_period', 'quantile', {'quantiles': (0.1, 0.9)})]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], **kwargs)

if __name__ == '__main__':
    process_oldcrow(year=2023, remove_last=False)



