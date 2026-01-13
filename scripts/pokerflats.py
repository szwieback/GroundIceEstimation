'''
Created on Oct 16, 2024

@author: simon
'''
import numpy as np
import pandas as pd
import datetime

from analysis import StefanPredictor, PredictionEnsemble, MulticlassPredictionEnsemble
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple, StefanStratigraphySmoothingSplineTalik)
from forcing import read_daily_noaa_forcing, parse_dates
from scripts.pathnames import paths
from scripts.plot_profile import read_InSAR
dist = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.40, 'low_horizon': 0.25, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.3, 'organic_below': 0.05},
    'n_factor': {'high': 0.95, 'low': 0.85, 'alphabeta': 2.0},
    'talik': {'low_depth': 0.3, 'high_depth': 0.5, 'probability': 0.5, 'high_thickness': 0.5,
              'low_thickness': 0.0, 'frozen_fraction': 0.1}}
ll, ur = None, None

geom = {'ia': 31.70 / 180 * np.pi}
datesstr = {2024: ('20240518', '20240530', '20240611', '20240623', '20240705', '20240717', '20240729',
                   '20240810', '20240822', '20240903', '20240915', '20240927')}
xy_ref = np.array([-147.4944, 65.1113])[:, np.newaxis]
wavelength = 0.055
var_atmo = (4e-3) ** 2
fns_unw_offset = {}



def pokerflats_forcing(fnforcing, year=2022):
    df = read_daily_noaa_forcing(fnforcing, convert_temperature=False)
    d0 = {2024: '2024-05-14'}[year]
    d1 = {2024: '2024-09-28'}[year]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())['T'].loc[pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    return dailytemp, ind_scenes

def process_pokerflats(year=2024, rmethod='hadamard', tmethod='none'):
    path0 = paths['stacks'] / f'Fairbanks_131_373/{year}/proc/{rmethod}/geocoded'
    fnforcing = paths['forcing'] / 'fairbanks/fairbanks.csv'
    pathout = paths['processed'] / f'pokerflats/{year}/{rmethod}_{tmethod}'

    N = 10000
    Nbatch = 1

    from analysis import (InversionProcessorIS, InversionResultsIS)

    s_obs, K, geospatial = read_InSAR(
        path0, wavelength, xy_ref=xy_ref, fns_unw_offset=fns_unw_offset.get(year, []), var_atmo=var_atmo,
        fill_nan=True)
    dailytemp, ind_scenes = pokerflats_forcing(fnforcing, year=year)
    indranges = [(ind_scenes[-5] + 5, ind_scenes[-1])] # from Aug 15

    predictor = StefanPredictor()
    
    Strat = StefanStratigraphySmoothingSplineTalik if tmethod == 'talik' else StefanStratigraphySmoothingSpline
    strat = StratigraphyMultiple(
        Strat(N=N, dist=dist), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    predens.predict_mean_period(indranges)

    data = {'s_obs': s_obs, 'K': K}
    for dname in data.keys():
        data[dname], geospatial_crop = geospatial.crop(data[dname], ll=ll, ur=ur)
    ip = InversionProcessorIS(predens, geospatial=geospatial_crop)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], pathout=pathout, n_jobs=-1, overwrite=True)
    ir.save(pathout / 'ir.p')
    ip.delete_temporary(pathout)
    ir = InversionResultsIS.from_file(pathout / 'ir.p')

    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
        ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e', 'quantile', {'quantiles': (0.1, 0.9)}),  ('e_mean_period', 'var'), ('e_mean_period', 'mean'),
        ('e_mean_period', 'quantile', {'quantiles': (0.1, 0.9)})]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], **kwargs)

if __name__ == '__main__':
    for tmethod in ('talik', 'none'):
        process_pokerflats(year=2024, tmethod=tmethod)
