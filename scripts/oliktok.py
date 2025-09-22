'''
Created on Oct 6, 2022

@author: simon
'''
import numpy as np
import pandas as pd
import datetime
import os
from pathlib import Path

from analysis import (StefanPredictor, PredictionEnsemble, enforce_directory, read_K, add_atmospheric_K, 
                      read_motion, InversionProcessorIS, InversionResultsIS, hdf5_attrs_from_tif,
                      export_defo_history_hdf5, get_dates_obs_str)

from simulation import StefanStratigraphySmoothingSpline, StratigraphyMultiple
from forcing import read_daily_noaa_forcing, parse_dates


geom = {'ia': 38.40 / 180 * np.pi}
wavelength = 0.055

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean': -3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.4, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},   # low_above 0.5/0.6/0.7
    # 'soil': {'high_horizon': 0.20, 'low_horizon': 0.15, 'organic_above': 0.1,
    #          'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'soil': {'high_horizon': 0.15, 'low_horizon': 0.10, 'organic_above': 0.12,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}


datesstr = {
    2023: ('20230603', '20230615', '20230627', '20230709', '20230721',
           '20230802', '20230814', '20230826', '20230907', '20230919'),
    2024: ('20240609', '20240621', '20240703', '20240715', '20240727',
           '20240808', '20240820', '20240901', '20240913', '20240925')
}

var_atmo = (4e-3) ** 2


def oliktok_forcing(fnforcing, year=2023, remove_last=True):
    df = read_daily_noaa_forcing(fnforcing, convert_temperature=False)
    d0 = {2023: '2023-05-20', 2024: '2024-05-20'}[year]
    d1 = {2023: '2023-09-30', 2024: '2024-09-30'}[year]
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    if remove_last: ind_scenes = ind_scenes[:-1]
    # dailytemp = (df.resample('D').mean())['T'][pd.date_range(start=d0, end=d1)]
    dailytemp = (df.resample('D').mean())[pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    return dailytemp, ind_scenes

def process_oliktok(year=2023, rmethod='hadamard', sensor='s1', remove_last=True):
    # path0 = f'/export/data/Experiments/stacks/OliktokPoint_P102D/{year}/proc/{rmethod}/geocoded'
    resolution = '40m'
    # resolution = '80m'
    fnforcing = '/export/data/Data/meteoro/forcing/Prudhoe/3925976.csv'
    pathout = Path(f'/export/data/Experiments/gie/processed/oliktok/{sensor}/{year}/{rmethod}/{resolution}_shallow_organic')

    if rmethod == 'hadamard':
        path0 = Path(f'/export/data/Experiments/stacks/oliktok/s1/P102D/stackpro/{year}/proc/hadamard/geocoded')
        fnunw = path0 / 'unwrapped.lonlat.tif'
        fnK = path0 / 'K_vec.lonlat.tif'
    else:
        path0 = Path(f'/export/data/Experiments/stacks/oliktok/s1/P102D/mintpy/outputs_{resolution}/{year}')
        fnunw = path0 / 'ph_history_resamp.tif'
        fnK = path0 / 'ph_history_cov_resamp.tif'

    N = 10000
    Nbatch = 1

    s_obs, geospatial = read_motion(fnunw, wavelength=wavelength)
    attrs = hdf5_attrs_from_tif(fnunw)

    attrs.pop('BAND_DESCRIPTIONS', None)
    K, geospatial_K = read_K(fnK)
    K = add_atmospheric_K(K, var_atmo)

    assert geospatial == geospatial_K
    print(f's_obs: {s_obs.shape}; K: {K.shape}')

    dailytemp, ind_scenes = oliktok_forcing(fnforcing, year=year, remove_last=False)
    dt_strlist = dailytemp.index.strftime('%Y-%m-%d').tolist()
    depth_list = [f'{i}-{i+2}' for i in range(0, 1500, 2)]

    print(f'daily temperature: {dailytemp.shape}, number of scenes: {len(ind_scenes)}')

    indranges = [(ind_scenes[-4], ind_scenes[-1])]  # from Aug 13

    predictor = StefanPredictor()
    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    predens.predict_mean_period(indranges)

    data = {'s_obs': s_obs, 'K': K}
    ip = InversionProcessorIS(predens, geospatial=geospatial)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], pathout=pathout, n_jobs=-1, overwrite=True)

    ir.save(os.path.join(pathout, 'ir.p'))
    ir = InversionResultsIS.from_file(os.path.join(pathout, 'ir.p'))

    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'),
        ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e_mean_period', 'var'), ('e_mean_period', 'mean'),
        ('e_mean_period', 'quantile', {'quantiles': (0.1, 0.9)})]
    export_defo_history_hdf5(
        data['s_obs'], pathout, geospatial_K, geom, K=data['K'],
        dates_obs_str=get_dates_obs_str(dailytemp, ind_scenes))

    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], hdf5=True, **kwargs)


if __name__ == '__main__':
    stack_method = 'mintpy'
    process_oliktok(year=2023, rmethod=stack_method, remove_last=False)
    # process_oliktok(year=2024, rmethod=stack_method, remove_last=False)


