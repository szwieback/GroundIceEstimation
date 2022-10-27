'''
Created on Sep 7, 2022

@author: simon
'''
import numpy as np
import pandas as pd
import datetime
import os

from analysis import StefanPredictor, PredictionEnsemble, enforce_directory
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple,
    StefanStratigraphyConstantE)
from forcing import load_forcing_merra_subset, parse_dates

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean': -3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.25, 'low_horizon': 0.10, 'organic_above': 0.1,
             'mineral_above': 0.05, 'mineral_below': 0.3, 'organic_below': 0.05},
    'n_factor': {'high': 0.95, 'low': 0.85, 'alphabeta': 2.0}}

def kivalina_forcing(folder_forcing, year=2019):
    df = load_forcing_merra_subset(folder_forcing)
    d0 = {2019: '2019-05-10', 2017: '2017-05-10', 2018: '2018-05-10'}[year]
    d1 = {2019: '2019-09-15', 2017: '2017-09-20', 2018: '2018-09-15'}[year]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())['T'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    datesstr = {2019:
             ('20190606', '20190618', '20190630', '20190712', '20190724', '20190805',
              '20190817', '20190829', '20190910')}
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    return dailytemp, ind_scenes

def process_kivalina(year=2019, rmethod='hadamard'):
    path0 = f'/10TBstorage/Work/stacks/Kivalina/gie/{year}/proc/{rmethod}/geocoded'
    folder_forcing = '/10TBstorage/Work/gie/forcing/kivalina'
    pathout = f'/10TBstorage/Work/gie/processed/kivalina/{year}/{rmethod}'
    # path0 = f'/home/simon/Work/gie/processed/kivalina/{year}/{rmethod}'
    # folder_forcing = '/home/simon/Work/gie/forcing/Kivalina'
    # pathout = os.path.join(path0, 'temp')
    geom = {'ia': 39.29 / 180 * np.pi}
    wavelength = 0.055
    var_atmo = (4e-3) ** 2
    xy_ref = np.array([-164.7300, 67.8586])[:, np.newaxis]
    # xy_ref = np.array([-164.79600, 67.8700])[:, np.newaxis]
    # ll, ur = (-164.8660, 67.8400), (-164.7185, 67.8600)
    ll, ur = (-164.8200, 67.8370), (-164.7185, 67.8600)
    
    N = 10000
    Nbatch = 1

    from analysis import (
        read_K, add_atmospheric, read_referenced_motion, InversionProcessor,
        InversionResults)

    fnunw = os.path.join(path0, 'unwrapped.geo.tif')
    fnK = os.path.join(path0, 'K_vec.geo.tif')
    K, geospatial_K = read_K(fnK)
    K = add_atmospheric(K, var_atmo)
    s_obs, geospatial = read_referenced_motion(fnunw, xy=xy_ref, wavelength=wavelength)
    assert geospatial == geospatial_K
    from analysis.ioput import save_geotiff
    
    # save_geotiff(s_obs - s_obs[4, ...][np.newaxis, ...], geospatial, os.path.join(pathout, 's_obs_late.tif'))
    
    dailytemp, ind_scenes = kivalina_forcing(folder_forcing, year=year)
    
    predictor = StefanPredictor()
    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)

    data = {'s_obs': s_obs, 'K': K}
    for dname in data.keys():
        data[dname], geospatial_crop = geospatial.crop(data[dname], ll=ll, ur=ur)
    ip = InversionProcessor(predens, geospatial=geospatial_crop)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], pathout=pathout, n_jobs=-1, overwrite=True)
    ir.save(os.path.join(pathout, 'ir.p'))
    ip.delete_weight_files(pathout)
    ir = InversionResults.from_file(os.path.join(pathout, 'ir.p'))
    
    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
        ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e', 'quantile', {'quantiles': (0.1, 0.9)})]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], **kwargs)

if __name__ == '__main__':
    process_kivalina()

