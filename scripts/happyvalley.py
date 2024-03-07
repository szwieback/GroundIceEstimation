'''
Created on Oct 6, 2022

@author: simon
'''
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

from analysis import StefanPredictor, PredictionEnsemble, MulticlassPredictionEnsemble
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple)
from forcing import read_daily_noaa_forcing, parse_dates

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean': -3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.4, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.20, 'low_horizon': 0.10, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}
ll, ur = (-148.8625, 69.1376), (-148.7590, 69.1640)

def happyvalley_forcing(fnforcing, year=2022):
    df = read_daily_noaa_forcing(fnforcing, convert_temperature=False)
    d0 = {2023: '2023-05-31', 2022: '2022-06-06', 2019: '2019-05-18'}[year]
    d1 = {2023: '2023-09-22', 2022: '2022-09-16', 2019: '2019-09-17'}[year]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())[pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    datesstr = {2023: ('20230605', '20230617', '20230629', '20230711', '20230723', '20230804',
                       '20230816', '20230828', '20230909'),
                2022: ('20220610', '20220622', '20220704', '20220716', '20220728',
                       '20220809', '20220821', '20220902', '20220914'),
                2019: ('20190602', '20190614', '20190626', '20190708', '20190720',
                       '20190801', '20190813', '20190825', '20190906')}
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    return dailytemp, ind_scenes

def process_happyvalley(year=2019, rmethod='hadamard'):
    path0 = Path(f'/10TBstorage/Work/stacks/Dalton_131_363/gie/{year}/proc/{rmethod}/geocoded')
    fnforcing = Path('/10TBstorage/Work/gie/forcing/sagwon/sagwon.csv')
    pathout = Path(f'/10TBstorage/Work/gie/processed/happyvalley/{year}/{rmethod}')

    fns_unw_offset = {2019: [(7, Path('/10TBstorage/Work/stacks/Dalton_131_363/2019_unw_offset.gpkg'))],
                      2022: [],
                      2023: []}[year]

    geom = {'ia': 38.40 / 180 * np.pi}
    wavelength = 0.055
    var_atmo = (4e-3) ** 2
    xy_ref = np.array([-148.8063, 69.1616])[:, np.newaxis]
    N = 10000
    Nbatch = 1

    from analysis import (
        read_K, add_atmospheric_K, read_referenced_motion, InversionProcessor,
        InversionResults)

    fnunw = path0 / 'unwrapped.geo.tif'
    fnK = path0 / 'K_vec.geo.tif'
    
    K, geospatial_K = read_K(fnK)
    s_obs, geospatial = read_referenced_motion(
        fnunw, xy=xy_ref, wavelength=wavelength, fns_unw_offset=fns_unw_offset)

    if year in (2019, 2022): # remove first acq because still a lot of snow
        K = K[1:, 1:, ...]
        s_obs = s_obs[1:, ...] - s_obs[0, ...][np.newaxis, ...]
    K = add_atmospheric_K(K, var_atmo)
    assert geospatial == geospatial_K

    dailytemp, ind_scenes = happyvalley_forcing(fnforcing, year=year)

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
    ir.save(pathout / 'ir.p')
    ip.delete_weight_files(pathout)
    ir = InversionResults.from_file(pathout / 'ir.p')

    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
        ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e', 'quantile', {'quantiles': (0.1, 0.9)})]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], **kwargs)

def process_happyvalley_ecotype(year=2019, rmethod='hadamard'):
    path0 = Path(f'/10TBstorage/Work/stacks/Dalton_131_363/gie/{year}/proc/{rmethod}/geocoded')
    p0 = Path('/10TBstorage/Work/gie')
    fnforcing = p0 / 'forcing/sagwon/sagwon.csv'
    pathout = p0 / f'processed/happyvalley/{year}/ecotype_{rmethod}'
    fnlc = p0 / f'ancillary/TNC/ecosystems_northern_alaska_jorgenson_2010.tif'

    fns_unw_offset = {2019: [(7, p0 / 'stacks/Dalton_131_363/2019_unw_offset.gpkg')],
                      2022: [],
                      2023: []}[year]
                      
    eclasses = {0: (1, 3, 11, 12, 13, 14, 15, 18, 23, 32, 41, 43, 44, 45, 46, 47, 48, 112, -99),
               1: (2, 21, 25, 26, 33, 34, 35)}        

    params_distribution_0 = params_distribution.copy()
    params_distribution_0['soil'] = {'high_horizon': 0.05, 'low_horizon': 0.00, 'organic_above': 0.1,
                                     'mineral_above': 0.3, 'mineral_below': 0.40, 'organic_below': 0.00}    
    multiclass_dist = {0: params_distribution_0, 1: params_distribution}                      

    geom = {'ia': 38.40 / 180 * np.pi}
    wavelength = 0.055
    var_atmo = (4e-3) ** 2
    xy_ref = np.array([-148.8063, 69.1616])[:, np.newaxis]
    N = 10000
    Nbatch = 1

    from analysis import (
        read_K, add_atmospheric_K, read_referenced_motion, InversionProcessor, MulticlassInversionResults)
    from scripts.ecotypes import reclassify
    
    fnunw = path0 / 'unwrapped.geo.tif'
    fnK = path0 / 'K_vec.geo.tif'
    
    K, geospatial_K = read_K(fnK)
    s_obs, geospatial = read_referenced_motion(
        fnunw, xy=xy_ref, wavelength=wavelength, fns_unw_offset=fns_unw_offset)

    if year in (2019, 2022): # remove first acq because still a lot of snow
        K = K[1:, 1:, ...]
        s_obs = s_obs[1:, ...] - s_obs[0, ...][np.newaxis, ...]
    K = add_atmospheric_K(K, var_atmo)
    assert geospatial == geospatial_K

    fnec = pathout / 'ec.tif'
    ec = reclassify(fnlc, eclasses, geospatial, fnout=fnec)

    dailytemp, ind_scenes = happyvalley_forcing(fnforcing, year=year)
    
    predictor = StefanPredictor()
    strats = {sc: StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N, dist=multiclass_dist[sc]), Nbatch=Nbatch) 
        for sc in multiclass_dist}
    predens = MulticlassPredictionEnsemble(strats, predictor, geom=geom)
    predens.predict(dailytemp)
    data = {'s_obs': s_obs, 'K': K, 'ec': ec[0, ...]}
    for dname in data.keys():
        data[dname], geospatial_crop = geospatial.crop(data[dname], ll=ll, ur=ur)
    print(geospatial_crop.shape)
    ip = InversionProcessor(predens, geospatial=geospatial_crop)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], ec=data['ec'], pathout=pathout, n_jobs=-1, overwrite=True)
    ir.save(pathout / 'ir.p')
    ip.delete_weight_files(pathout)
    
    ir = MulticlassInversionResults.from_file(pathout / 'ir.p')

    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
        ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e', 'quantile', {'quantiles': (0.1, 0.9)})]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], **kwargs)

if __name__ == '__main__':
    # process_happyvalley(year=2019)
    # process_happyvalley(year=2022)
    process_happyvalley_ecotype(year=2022)


