import numpy as np
from pathlib import Path

from analysis import (
    StefanPredictor, PredictionEnsemble, read_referenced_InSAR, InversionProcessorIS, get_dates_obs_str,
    InversionResultsISMmap, MulticlassInversionResultsISMmap, export_defo_history_hdf5, read_meta_from_json,
    )
from simulation import StefanStratigraphySmoothingSpline
from forcing import forcing_merra_meta, forcing_daily_noaa_meta
from scripts.pathnames import paths

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean': -3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.7, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.20, 'low_horizon': 0.15, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}

def process_mintpy(
        sitename, year, pmintpy, pforcing, params_distribution, ecotype=False, xy_ref=None,
        overwrite=False, N=10000, Tf=None):

    ecotypename = 'singleensemble' if not ecotype else 'ecotype'
    pathout = paths['processed'] / f'{sitename}/{year}/{ecotypename}'
    fnunw = pmintpy / 'ph_history.tif'
    fnK = pmintpy / 'ph_history_cov.tif'
    fnmeta = pmintpy / 'meta.json'
    
    meta = read_meta_from_json(fnmeta)
    # dailytemp, ind_scenes = forcing_merra_meta(pforcing, meta, year=year)
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
    export_defo_history_hdf5(
        data['s_obs'], pathout, geospatial, meta['geom'], K=data['K'],
        dates_obs_str=get_dates_obs_str(dailytemp, ind_scenes))
    
    predens.predict(dailytemp)
    ip = IP(predens, geospatial=geospatial, blocksize=512, **kwargs)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], ec=data['ec'], pathout=pathout, n_jobs=-1, memory=False,
        overwrite=overwrite)
    ir.save(pathout / 'ir.p')
    ip.delete_temporary(pathout)
    ir = IR.from_file(pathout / 'ir.p')

    ir.register_dates(dailytemp.index)  # for h5 export
    # qdict = {'quantiles': (0.1, 0.9)}#('e', 'quantile', qdict)
    expecs = [('yf', 'mean'), ('e', 'mean'), ('e', 'var'), ]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(
            pathout, param=expec[0], etype=expec[1], hdf5=True, overwrite=overwrite, **kwargs)

if __name__ == '__main__':

    sitename = 'utqiagvik2'
    # 2023 has major unwrapping issues
    for year in [2023, 2024]:#[2024]:
        if year == 2024:
            xy_ref = np.array([[575172.0, 7905233.4]]).T
        elif year == 2023:
            xy_ref = np.array([[577547.0, 7906647.0]]).T        
        ecotype = False
        pmintpy = Path(f'/10TBstorage/Work/MintPy/utqiagvik2/s1/73/{year}')
        pforcing = Path('/10TBstorage/Work/MintPy/utqiagvik2/3923298.csv')
        process_mintpy(
            sitename, year, pmintpy, pforcing, params_distribution, ecotype=ecotype, xy_ref=xy_ref, 
            overwrite=True)
