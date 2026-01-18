'''
Created on Dec 2, 2025

@author: simon
'''
import numpy as np
from pathlib import Path

from analysis import (
    StefanPredictor, PredictionEnsemble, read_referenced_InSAR, InversionProcessorIS,
    InversionResultsISMmap, MulticlassInversionResultsISMmap, read_meta_from_json,
    )
from simulation import StefanStratigraphySmoothingSpline
from forcing import forcing_daily_noaa_meta
from scripts.pathnames import paths

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean': -3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.7, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.20, 'low_horizon': 0.15, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}

def process_mintpy_downscaled(
        sitename, year, pmintpy, pforcing, params_distribution, ecotype=False, xy_ref=None,
        overwrite=False, N=10000, Tf=None):

    downscale = 7
    oversample_factor = 2

    ecotypename = 'singleensemble' if not ecotype else 'ecotype'
    pathout = paths['processed'] / f'{sitename}/{year}/{ecotypename}'
    fnunw_raw = pmintpy / 'ph_history.tif'
    fnK_raw = pmintpy / 'ph_history_cov.tif'
    fnmeta = pmintpy / 'meta.json'

    meta = read_meta_from_json(fnmeta)
    dailytemp, ind_scenes = forcing_daily_noaa_meta(pforcing, meta, year=year, convert_temperature=False)

    
    # convert to utm
    fnunw_hr, fnK_hr = pmintpy / 'ph_history_utm_hr.tif', pmintpy / 'ph_history_cov_utm_hr.tif'
    from analysis import Geospatial, save_geotiff
    geospatial_raw = Geospatial.from_file(fnunw_raw)
    geospatial_utm_hr = geospatial_raw.to_utm()
    geospatial_K = Geospatial.from_file(fnK_raw)

    # high-resolution utm
    unw_hr, _ = geospatial_utm_hr.warp_from_file(fnunw_raw)
    save_geotiff(unw_hr, geospatial_utm_hr, fnunw_hr)
    K_hr, _ = geospatial_utm_hr.warp_from_file(fnK_raw)
    from warnings import warn
    warn("Applying scale factor to deal with MintPy bug (not fixed, but export through MintPyAPI for this dataset did not deal with it.")
    K_hr *= meta['wavelength'] / (4 * np.pi)
    save_geotiff(K_hr, geospatial_utm_hr, fnK_hr)
    del unw_hr, K_hr

    # downscaled utm, accounting for oversampling when computing K
    fnunw, fnK = pmintpy / 'ph_history_utm.tif', pmintpy / 'ph_history_cov_utm.tif'
    geospatial = geospatial_utm_hr.downscaled(downscale=downscale)
    unw, _ = geospatial.warp_from_file(fnunw_raw, method='average')
    save_geotiff(unw, geospatial, fnunw)
    K, _ = geospatial.warp_from_file(fnK_raw, method='average')
    K = K  * (downscale / oversample_factor) ** (-2)
    save_geotiff(K, geospatial, fnK)

    if xy_ref is not None:
        # data, geospatial = read_referenced_InSAR(
        #     fnunw, fnK, xy_ref, wavelength=meta['wavelength'], fndist=pathout / 'distance_cal.p',
        #     fnunw_hr=fnunw_hr, fnK_hr=fnK_hr, overwrite=overwrite)
        data, geospatial = read_referenced_InSAR(
            fnunw, fnK, xy_ref, wavelength=meta['wavelength'], fndist=pathout / 'distance_cal.p',
            overwrite=overwrite)        
    else:
        raise NotImplementedError("xy_ref needed")

    if not ecotype:
        ec = None
    else:
        raise NotImplementedError("ecotype input requires special consideration")

    IP = InversionProcessorIS
    if not ecotype:
        IR = InversionResultsISMmap
    else:
        IR = MulticlassInversionResultsISMmap
    kwargs = {}

    predictor = StefanPredictor()

    strat = StefanStratigraphySmoothingSpline(N=N, dist=params_distribution)
    if Tf is not None:
        strat.ancillary['Tf'] = Tf
    predens = PredictionEnsemble(strat, predictor, geom=meta['geom'])
    data['ec'] = ec

    # store an atmospherically referenced deformation history
    geospatial.save_geotiff(data['s_obs'], pathout / 's_obs.tif')
    
    predens.predict(dailytemp)
    ip = IP(predens, geospatial=geospatial, blocksize=512, **kwargs)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], ec=data['ec'], pathout=pathout, n_jobs=-1, memory=False,
        overwrite=overwrite)
    ir.save(pathout / 'ir.p')
    ip.delete_temporary(pathout)
    ir = IR.from_file(pathout / 'ir.p')

    # qdict = {'quantiles': (0.1, 0.9)}#('e', 'quantile', qdict)
    expecs = [('yf', 'mean'), ('e', 'mean'), ('e', 'var'), ]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(
            pathout, param=expec[0], etype=expec[1], hdf5=True, overwrite=overwrite, **kwargs)


def process_mintpy_old(
        sitename, year, pmintpy, pforcing, params_distribution, ecotype=False, xy_ref=None,
        overwrite=False, N=10000, Tf=None):

    ecotypename = 'singleensemble' if not ecotype else 'ecotype'
    pathout = paths['processed'] / f'{sitename}/{year}/{ecotypename}'
    fnunw_raw = pmintpy / 'ph_history.tif'
    fnK_raw = pmintpy / 'ph_history_cov.tif'
    fnmeta = pmintpy / 'meta.json'
    
    # convert to utm
    fnunw, fnK = pmintpy / 'ph_history_utm.tif', pmintpy / 'ph_history_cov_utm.tif'
    from analysis import Geospatial, save_geotiff
    geospatial = Geospatial.from_file(fnunw_raw)
    geospatial_utm = geospatial.to_utm()
    geospatial_K = Geospatial.from_file(fnK_raw)

    unw, _ = geospatial_utm.warp_from_file(fnunw_raw)
    save_geotiff(unw, geospatial_utm, fnunw)
    K, _ = geospatial_utm.warp_from_file(fnK_raw)
    save_geotiff(K, geospatial_utm, fnK)
    
    meta = read_meta_from_json(fnmeta)
    dailytemp, ind_scenes = forcing_daily_noaa_meta(pforcing, meta, year=year, convert_temperature=False)

    
    if xy_ref is not None:
        data, geospatial = read_referenced_InSAR(
            fnunw, fnK, xy_ref, wavelength=meta['wavelength'], fndist=pathout / 'distance_cal.p',
            overwrite=True)
    else:
        raise NotImplementedError("xy_ref needed")

    if not ecotype:
        ec = None
    else:
        raise NotImplementedError("ecotype input requires special consideration")
    
    IP = InversionProcessorIS
    if not ecotype:
        IR = InversionResultsISMmap
    else:
        IR = MulticlassInversionResultsISMmap
    kwargs = {}
    
    predictor = StefanPredictor()

    strat = StefanStratigraphySmoothingSpline(N=N, dist=params_distribution)
    if Tf is not None:
        strat.ancillary['Tf'] = Tf
    predens = PredictionEnsemble(strat, predictor, geom=meta['geom'])
    data['ec'] = ec

    # store an atmospherically referenced deformation history
    geospatial.save_geotiff(data['s_obs'], pathout / 's_obs.tif')
    
    predens.predict(dailytemp)
    ip = IP(predens, geospatial=geospatial, blocksize=512, **kwargs)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], ec=data['ec'], pathout=pathout, n_jobs=-1, memory=False,
        overwrite=overwrite)
    ir.save(pathout / 'ir.p')
    ip.delete_temporary(pathout)
    
    ir = IR.from_file(pathout / 'ir.p')

    # qdict = {'quantiles': (0.1, 0.9)}#('e', 'quantile', qdict)
    expecs = [('yf', 'mean'), ('e', 'mean'), ('e', 'var'), ]
    
    ir._save_memmap_to_npy(pathout / 'e_mean.dat', pathout / 'e_mean.npy')
    expecs = [('e', 'var'), ]
    
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(
            pathout, param=expec[0], etype=expec[1], hdf5=True, overwrite=overwrite, **kwargs)

if __name__ == '__main__':

    sitename = 'utqiagvik'
    year = 2024
    # xy_ref = np.array([[7904132.4, 574612.0], [7911619.2, 579293.5], [7904588.9, 587598.4]]).T
    # xy_ref = np.array([[573445.8, 7902347.9],  [582798.6, 7915368.0], [588067.5, 7905825.9]]).T #[582723.1, 7915281.6],

    # xy_ref = np.array([[573445.8, 7902347.9], [583009.8, 7915563.0], [588067.5, 7905825.9]]).T # full resolution
    # xy_ref = np.array([[573445.8, 7902347.9], [581455.2, 7914007.1], [588067.5, 7905825.9]]).T #okay
    # xy_ref = np.array([[573509.7, 7902352.7], [581398.4, 7913988.2], [588067.5, 7905825.9]]).T  # okay
    # reprocessed data
    # xy_ref = np.array([[580349.2, 7911352.1], [587462.7, 7906246.4]]).T
    xy_ref = np.array([[583008.7, 7915513.2], [574034.2, 7903403.5], [587557.5, 7904630.7]]).T#[579849.3, 7911152.9]]).T
    xy_ref = np.array([[583008.7, 7915513.2], [575643.0, 7905235.0], [587557.5, 7904630.7], [580360.6, 7911357.0]]).T
    xy_ref = np.array([[583008.7, 7915513.2], [576948.2, 7906320.7], [587605.0, 7904553.6], [580360.6, 7911357.0]]).T

    

    ecotype = False
    pmintpy = Path(f'/10TBstorage/Work/MintPy/utqiagvik/alos2/{year}')
    pforcing = Path('/10TBstorage/Work/MintPy/utqiagvik2/3923298.csv')
    process_mintpy_downscaled(
        sitename, year, pmintpy, pforcing, params_distribution, ecotype=ecotype, xy_ref=xy_ref, 
        overwrite=True)
