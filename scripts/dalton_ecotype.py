#!/usr/bin/env python
'''
Created on Oct 6, 2022

@author: simon
'''

import os
import numpy as np
import pandas as pd
import datetime
import time

from analysis import (
    StefanPredictor, PredictionEnsemble, MulticlassPredictionEnsemble, read_K,
    RationalQuadraticSepDiagCovMV, add_nugget, read_geotiff_geospatial, assemble_tril,
    spatial_referencing, length_conversion, InversionProcessorIS, InversionResultsISMmap,
    InversionResultsIS, InversionProcessorGM, InversionResultsGM, InversionResultsGMMmap,
    MulticlassInversionResultsIS, MulticlassInversionResultsISMmap, MulticlassInversionResultsGM,
    MulticlassInversionResultsGMMmap, get_dates_obs_str, export_defo_history_hdf5)
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple)
from forcing import read_daily_noaa_forcing, parse_dates
from scripts.pathnames import paths

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.4, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.25, 'low_horizon': 0.15, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}

eclasses = {0: (1, 3, 11, 12, 13, 14, 15, 18, 23, 32, 41, 43, 44, 45, 46, 47, 48, 112, -99),
           1: (2, 21, 25, 26, 33, 34, 35)}

params_distribution_0 = params_distribution.copy()
params_distribution_0['soil'] = {'high_horizon': 0.05, 'low_horizon': 0.00, 'organic_above': 0.1,
                                 'mineral_above': 0.3, 'mineral_below': 0.40, 'organic_below': 0.00}
multiclass_dist = {0: params_distribution_0, 1: params_distribution}

def dalton_forcing(fnforcing, year=2022):
    df = read_daily_noaa_forcing(fnforcing, convert_temperature=False)
    d0 = {2023: '2023-05-31', 2022: '2022-06-06', 2019: '2019-05-18'}[year]
    # d1 = {2023: '2023-09-22', 2022: '2022-09-16', 2019: '2019-09-17'}[year]
    d1 = {2023: '2023-09-25', 2022: '2022-09-16', 2019: '2019-09-25'}[year]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())[pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    datesstr = {2019: ('20190602', '20190614', '20190626', '20190708', '20190720', '20190801',
                       '20190825', '20190906', '20190918'),
                2023: ('20230605', '20230617', '20230629', '20230723', '20230711',
                       '20230804', '20230828', '20230909', '20230921')}
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    return dailytemp, ind_scenes

def process_dalton(
        xy_ref, ecotype=False, year=2019, imethod='IS', rmethod='mintpy', memory=True, K_value=2,
        N=10000, overwrite=True):
    fnforcing = paths['forcing'] / 'sagwon/sagwon.csv'
    site_name = 'dalton'
    ecotypename = 'singleensemble' if not ecotype else 'ecotype'
    if imethod in ['IS', 'IS_full']:
        pathout = paths['processed'] / f'{site_name}/{year}/{ecotypename}/{rmethod}_{imethod}'
    else:
        pathout = paths['processed'] / f'{site_name}/{year}/{ecotypename}/{rmethod}_{imethod}_K{K_value}'

    pathdefo = paths['stacks'] / f'{site_name}/{year}'
    fnunw = pathdefo / 'ph_history.tif'
    fnK = pathdefo / 'ph_history_cov.tif'

    geom = {'ia': 38.40 / 180 * np.pi}
    wavelength = 0.055
    Nbatch = 1

    from scripts.kivalina_calibration import caldict
    unw, geospatial_unw = read_geotiff_geospatial(fnunw)
    K, geospatial_K = read_K(fnK)

    P = K.shape[0] + 1
    var_atmo = np.ones(P) * (caldict['var_rad'])  # in rad
    covmodel = RationalQuadraticSepDiagCovMV(caldict['l'], var_atmo, alpha=caldict['alpha'])
    K = add_nugget(K, caldict['nugget_speckle'])

    fndist = pathout / 'distance_cal.p'
    unw_cor, K_cor = spatial_referencing(
        unw, K, covmodel, xy_ref, geospatial_K, fndist=fndist, convert_to_length=False, overwrite=overwrite)
    s_obs, K_s = length_conversion(unw_cor, K_cor, wavelength=wavelength, flip_sign=True)

    if ecotype:
        from scripts.ecotypes import reclassify
        fnlc = paths['ancillary'] / 'TNC/ecosystems_northern_alaska_jorgenson_2010.tif'
        fnec = pathout / 'ec.tif'
        ec = reclassify(fnlc, eclasses, geospatial_K, fnout=fnec, overwrite=overwrite)
    else:
        ec = None

    dailytemp, ind_scenes = dalton_forcing(fnforcing, year=year)

    indranges = [(ind_scenes[-4], ind_scenes[-1])]
    depthranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    if imethod in ('IS', 'IS_full'):
        IP = InversionProcessorIS
        if not ecotype:
            IR = InversionResultsIS if memory else InversionResultsISMmap
        else:
            IR = MulticlassInversionResultsIS if memory else MulticlassInversionResultsISMmap
        kwargs = {}
    elif imethod == 'GM':
        IP = InversionProcessorGM
        if not ecotype:
            IR = InversionResultsGM if memory else InversionResultsGMMmap
        else:
            IR = MulticlassInversionResultsGM if memory else MulticlassInversionResultsGMMmap
        kwargs = {'K': K_value,
            'variables': (('e', {'indranges': indranges}),
                          ('e', {'depthranges': depthranges}),
                          ('yf', {'ind_scene': [(ind_scenes[-1])]}),
                          )}

    predictor = StefanPredictor()
    if ecotype:
        strats = {sc: StratigraphyMultiple(
            StefanStratigraphySmoothingSpline(N=N, dist=multiclass_dist[sc]), Nbatch=Nbatch)
            for sc in multiclass_dist}
        ec = ec[0, ...]
        predens = MulticlassPredictionEnsemble(strats, predictor, geom=geom)
    else:
        strat = StratigraphyMultiple(
            StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)
        predens = PredictionEnsemble(strat, predictor, geom=geom)
        ec = None
    data = {'s_obs': s_obs, 'K': np.moveaxis(assemble_tril(np.moveaxis(K_s, 0, -1)), (0, 1), (-2, -1)),
            'ec': ec}
    export_defo_history_hdf5(
        data['s_obs'], pathout, geospatial_K, geom, K=data['K'],
        dates_obs_str=get_dates_obs_str(dailytemp, ind_scenes))
    raise
    predens.predict(dailytemp)
    predens.predict_mean_depth(depthranges)
    predens.predict_mean_period(indranges)
    ip = IP(predens, geospatial=geospatial_K, blocksize=512, **kwargs)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], ec=data['ec'], pathout=pathout, n_jobs=-1, memory=memory,
        overwrite=True)
    ir.save(pathout / 'ir.p')
    ip.delete_temporary(pathout)
    ir = IR.from_file(pathout / 'ir.p')

    ir.register_dates(dailytemp.index)  # for h5 export
    qdict = {'quantiles': (0.1, 0.9)}
    expecs = [('e_mean_period', 'mean'), ('yf', 'mean'), ('e_mean_depth', 'mean'),
              ('e_mean_period', 'var'), ('e_mean_depth', 'var'), ('e_mean_period', 'quantile', qdict),
              ('e_mean_depth', 'quantile', qdict)]

    if imethod == 'IS_full':
        expecs = expecs + [('e', 'mean'), ('e', 'var'), ('e', 'quantile', qdict)]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(
            pathout, param=expec[0], etype=expec[1], hdf5=True, overwrite=overwrite, **kwargs)

if __name__ == '__main__':


    year = 2023
    do_gmi = False
    do_isi = False
    do_isi_full = True
    rmethod = 'mintpy'
    N = 50000
    memory = False
    xy_ref = np.array([
                    [430013.0, 7679097.7],  [427668.9, 7656177.3]]).T
    for ecotype in (True,):
        if do_gmi:
            imethod = 'GM'
            for K_value in (1, 2, 3, 4, 5):
                start_is = time.time()
                process_dalton(
                    xy_ref, N=N, imethod=imethod, memory=memory, year=year, rmethod=rmethod,
                    K_value=K_value, ecotype=ecotype)
                end_is = time.time()
                t = end_is - start_is
                print(f"Runtime for {imethod} method, K={K_value}: {t:.2f} seconds")

        if do_isi:
            imethod = 'IS'
            start_is = time.time()
            process_dalton(
                xy_ref, N=N, imethod=imethod, memory=memory, year=year, rmethod=rmethod, ecotype=ecotype)
            end_is = time.time()
            t = end_is - start_is
            print(f"Runtime for {imethod} method: {t:.2f} seconds")
        if do_isi_full:
            imethod = 'IS_full'
            N = 10000
            process_dalton(
                xy_ref, N=N, imethod=imethod, memory=memory, year=year, rmethod=rmethod, ecotype=ecotype)


