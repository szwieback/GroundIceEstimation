'''
Created on Oct 6, 2022

@author: simon
'''
import numpy as np
import pandas as pd
import datetime
import os
from pathlib import Path

from analysis import StefanPredictor, PredictionEnsemble, enforce_directory
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple,
    StefanStratigraphyConstantE)
from forcing import read_daily_noaa_forcing, parse_dates

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean': -3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.4, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.20, 'low_horizon': 0.15, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}
# ll, ur = (-156.9, 71.25), (-156.6, 71.35)
ll, ur = (-156.78, 71.28), (-156.67, 71.32)

datesstr = {
    2023: ('20230613', '20230625', '20230707', '20230719', '20230731',
           '20230812', '20230824', '20230905', '20230917', '20230929'),
    2024: ('20240619', '20240701', '20240713', '20240725',
        '20240818', '20240830', '20240911', '20240923')
}

def utqiagvik_forcing(fnforcing, year=2023, remove_last=True):
    df = read_daily_noaa_forcing(fnforcing, convert_temperature=False)
    d0 = {2023: '2023-05-20', 2024: '2024-05-16'}[year]
    d1 = {2023: '2023-09-30', 2024: '2024-10-07'}[year]
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    if remove_last: ind_scenes = ind_scenes[:-1]
    # dailytemp = (df.resample('D').mean())['T'][pd.date_range(start=d0, end=d1)]
    dailytemp = (df.resample('D').mean())[pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    return dailytemp, ind_scenes

def process_utqiagvik(year=2023, rmethod='hadamard', sensor='s1', remove_last=True):
        # path0 = f'/export/data/Experiments/stacks/OliktokPoint_P102D/{year}/proc/{rmethod}/geocoded'
    fnforcing = Path('/export/data/Data/meteoro/forcing/Barrow_airport/3923298.csv')
    pathout = Path(f'/export/data/Experiments/gie/processed/utqiagvik/{sensor}/{year}/{rmethod}')

    if rmethod == 'hadamard':
        path0 = f'/export/data/Experiments/stacks/utqiagvik/s1/P73D/stackpro/{year}/proc/hadamard/geocoded'
        fnunw = os.path.join(path0, 'unwrapped.lonlat.tif')
        fnK = os.path.join(path0, 'K_vec.lonlat.tif')
    else:
        path0 = Path(f'/export/data/Experiments/stacks/utqiagvik/s1/P73D/mintpy/outputs/{year}')
        fnunw = path0 / 'ph_history.lonlat.tif'
        fnK = path0 / 'ph_history_cov.lonlat.tif'

    # path0 = f'/export/data/Experiments/stacks/utqiagvik_P73D/{year}/proc/{rmethod}/geocoded'
    # fnforcing = '/export/data/Data/meteoro/forcing/Barrow_airport/3923298.csv'
    # pathout = f'/export/data/Experiments/gie/processed/Utqiagvik/{year}/{rmethod}'

    # fns_unw_offset = {2019: [(7, os.path.join('/10TBstorage/Work/stacks/Dalton_131_363/2019_unw_offset.gpkg'))],
    #                   2022: []}[year]

    geom = {'ia': 38.40 / 180 * np.pi}
    wavelength = 0.055
    var_atmo = (4e-3) ** 2
    # xy_ref = np.array([-156.7723, 71.2825])[:, np.newaxis]
    N = 10000
    Nbatch = 1

    from analysis import (
        read_K, add_atmospheric_K, read_referenced_motion, InversionProcessorIS,
        InversionResultsIS)
    from analysis import ioput

    # fnunw = os.path.join(path0, 'unwrapped.geo.tif')
    # fnK = os.path.join(path0, 'K_vec.geo.tif')
    
    K, geospatial_K = read_K(fnK)
    K = add_atmospheric_K(K, var_atmo)
    # s_obs, geospatial = read_referenced_motion(
    #     fnunw, xy=xy_ref, wavelength=wavelength, fns_unw_offset=fns_unw_offset)
    # s_obs, geospatial = read_referenced_motion(fnunw, xy=xy_ref, wavelength=wavelength)
    s_obs, geospatial = ioput.read_motion(fnunw, wavelength=wavelength)

    # if year in (2023, 2024): # remove first acq because still a lot of snow
    #     K = K[1:, 1:, ...]
    #     s_obs = s_obs[1:, ...] - s_obs[0, ...][np.newaxis, ...]

    assert geospatial == geospatial_K
    print(f's_obs: {s_obs.shape}; K: {K.shape}')

    dailytemp, ind_scenes = utqiagvik_forcing(fnforcing, year=year, remove_last=False)
    print(f'daily temperature: {dailytemp.shape}, number of scenes: {len(ind_scenes)}')

    indranges = [(ind_scenes[-4], ind_scenes[-1])]  # from Aug 13

    predictor = StefanPredictor()
    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    predens.predict_mean_period(indranges)

    if remove_last:
        s_obs = s_obs[:-1, ...]
        K = K[:-1,:-1, ...]
    data = {'s_obs': s_obs, 'K': K}
    for dname in data.keys():
        data[dname], geospatial_crop = geospatial.crop(data[dname], ll=ll, ur=ur)

    print(f'cropped s_obs: {data['s_obs'].shape}; K: {data['K'].shape}')
    ip = InversionProcessorIS(predens, geospatial=geospatial_crop)

    # ip = InversionProcessor(predens, geospatial=geospatial)

    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], pathout=pathout, n_jobs=-1, overwrite=True)
    ir.save(os.path.join(pathout, 'ir.p'))
    # ip.delete_weight_files(pathout)
    ir = InversionResultsIS.from_file(os.path.join(pathout, 'ir.p'))

    # expecs = [
    #     ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
    #     ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
    #     ('e', 'quantile', {'quantiles': (0.1, 0.9)})]
    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
        ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e_mean_period', 'var'), ('e_mean_period', 'mean'),
        ('e_mean_period', 'quantile', {'quantiles': (0.1, 0.9)})]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], **kwargs)

if __name__ == '__main__':
    # stack_method = 'hadamard'
    stack_method = 'mintpy'
    process_utqiagvik(year=2023, rmethod=stack_method, remove_last=False)
    process_utqiagvik(year=2024, rmethod=stack_method, remove_last=False)


