'''
Created on Jun 16, 2023

@author: simon
'''
import pandas as pd
import numpy as np
import os
import datetime

from forcing import load_forcing_merra_subset, parse_dates, ind_TDD_exceedance
from analysis import (save_object, load_object, read_K, read_geotiff_geospatial, 
    RationalQuadraticSepDiagCovMV, add_nugget, length_conversion, spatial_referencing, InversionProcessor,
    InversionResults, StefanPredictor, PredictionEnsemble, assemble_tril)
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple)

wavelength = 0.055   

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.4, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.20, 'low_horizon': 0.10, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}

def kivalina_forcing(folder_forcing, year, remove_last=True):
    df = load_forcing_merra_subset(folder_forcing)
    d0 = {2019: '2019-05-14', 2018: '2018-05-18'}[year]
    d1 = {2019: '2019-09-15', 2018: '2018-09-15'}[year]
    datesstr = {2019:
                 ('20190606', '20190618', '20190630', '20190712', '20190724', '20190805',
                  '20190817', '20190829', '20190910'),
                2018:
                 ('20180611', '20180623', '20180705', '20180717', '20180729', '20180810', '20180822', 
                  '20180903')}
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    if remove_last: ind_scenes = ind_scenes[:-1]    
    dailytemp = (df.resample('D').mean())['T'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    return dailytemp, ind_scenes

def indranges_kivalina(dailytemp, ind_scenes, TDD=[900, 1000]):
    inds0 = ind_TDD_exceedance(dailytemp, TDD)
    inds0n = [f'TDD{tdd}' for tdd in TDD]
    inds1 = (len(dailytemp) - 1, ind_scenes[-1])
    inds1n = ['lastday', 'lastscene']
            
    indranges_dict = {f'{ind0n}_{ind1n}': (ind0, ind1) 
                      for ind0, ind0n in zip(inds0, inds0n) for ind1, ind1n in zip(inds1, inds1n)}
    
    return indranges_dict

def process_index_kivalina(
        TDD, geom, pathin, pathout, folder_forcing, fnref, wavelength=wavelength, N=10000, Nbatch=1, 
        year=2019, remove_last=False, overwrite=True):
    dailytemp, ind_scenes = kivalina_forcing(folder_forcing, year, remove_last=remove_last)
    indranges_dict = indranges_kivalina(dailytemp, ind_scenes, TDD=TDD)
    indranges_names, indranges = tuple(indranges_dict.keys()), tuple(indranges_dict.values())
    
    # save forcing and timing info
    dict_forcing = {
        'dailytemp': dailytemp, 'ind_scenes': ind_scenes, 'indranges_names': indranges_names, 
        'indranges': indranges}
    save_object(dict_forcing, os.path.join(pathout, 'forcing_timing.p'))
    
    _fununw = 'unwrapped_corr.geo.tif' if year == 2019 else 'unwrapped.geo.tif'
    fnunw = os.path.join(pathin, _fununw)
    fnK = os.path.join(pathin, 'K_vec.geo.tif')
    K, geospatial_K = read_K(fnK)
    unw, geospatial_unw = read_geotiff_geospatial(fnunw)
    assert geospatial_unw == geospatial_K    
    
    from scripts.kivalina_calibration import caldict
    if remove_last:
        unw = unw[:-1, ...]
        K = K[:-1, :-1, ...]        
    P = K.shape[0] + 1
    var_atmo = np.ones(P) * (caldict['var_rad'])  # in rad
    # initialize covariancemodel
    covmodel = RationalQuadraticSepDiagCovMV(caldict['l'], var_atmo, alpha=caldict['alpha'])
    # apply nugget
    K = add_nugget(K, caldict['nugget_speckle'])
    
    # todo: add option for only ref 10 
    xy_ref = load_object(fnref)['regular']
    fndist = os.path.join(pathout, 'distance_cal.p')
    unw_cor, K_cor = spatial_referencing(
        unw, K, covmodel, xy_ref, geospatial_K, fndist=fndist, convert_to_length=False, overwrite=overwrite)
    # to motion
    s_obs, K_s = length_conversion(unw_cor, K_cor, wavelength=wavelength, flip_sign=True)
    
    # check whether 2018 needs different references
    predictor = StefanPredictor()
    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    predens.predict_mean_period(indranges)
    
    data, geospatial_crop = {'s_obs': s_obs, 'K': K_s}, geospatial_unw
    # # for testing only
    # ll, ur = (-164.8200, 67.8370), (-164.7800, 67.8450)    
    # for dname in data:
    #     data[dname], geospatial_crop = geospatial_unw.crop(data[dname], ll=ll, ur=ur)
    ip = InversionProcessor(predens, geospatial=geospatial_crop)
    _K = np.moveaxis(assemble_tril(np.moveaxis(data['K'], 0, -1)), (0, 1), (-2, -1))
    ir = ip.results(
        ind_scenes, data['s_obs'], _K, pathout=pathout, n_jobs=-1, overwrite=overwrite)
    ir.save(os.path.join(pathout, 'ir.p'))
    raise
    ip.delete_weight_files(pathout)
    ir = InversionResults.from_file(os.path.join(pathout, 'ir.p'))

    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
        ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e', 'quantile', {'quantiles': (0.1, 0.9)}),
        ('e_mean_period', 'mean'), ('e_mean_period', 'var'),
        ('e_mean_period', 'quantile', {'quantiles': (0.1, 0.9)})]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], **kwargs)

if __name__ == '__main__':
    scenarios = {'2019': (2019, False), '2019r': (2019, True), '2018': (2018, False)}
    TDD = (850, 950)
    N, Nbatch = 10000, 1    
    geom = {'ia': 39.29 / 180 * np.pi}
    
    path0 = '/10TBstorage/Work/gie'
    folder_forcing = os.path.join(path0, 'forcing', 'kivalina')
    pathin0 = '/10TBstorage/Work/stacks/Kivalina/gie/'
    pathout0 = os.path.join(path0, 'processed', 'kivalina', 'index')
    
    fnref= 'references_latlon.p' #adjust this with only 10th ref, rename out
    
    overwrite = False
    # loop over scenarios

    scenario = '2019r'
    year, remove_last = scenarios[scenario]
    pathout = os.path.join(pathout0, scenario)
    pathin = os.path.join(pathin0, str(year), 'proc', 'hadamard', 'geocoded')
    fnref_full = os.path.join(pathin, fnref)
    process_index_kivalina(
        TDD, geom, pathin, pathout, folder_forcing, fnref_full, wavelength=wavelength, N=N, Nbatch=Nbatch, 
        year=year, remove_last=remove_last, overwrite=overwrite)

        
        