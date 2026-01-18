'''
Created on Oct 31, 2025

@author: simon
'''
import numpy as np
from pathlib import Path

from analysis import (
    StefanPredictor, PredictionEnsemble, read_referenced_InSAR, InversionProcessorIS,
    InversionResultsISMmap, MulticlassInversionResultsISMmap, read_meta_from_json,
    MulticlassPredictionEnsemble)
from simulation import StefanStratigraphySmoothingSpline
from forcing import forcing_merra_meta
from scripts.pathnames import paths

params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.7, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.30, 'low_horizon': 0.20, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}

eclasses = {0: (1, 3, 11, 12, 13, 14, 15, 18, 23, 32, 41, 43, 44, 45, 46, 47, 48, 112, -99),
           1: (2, 21, 25, 26, 33, 34, 35)}

params_distribution_0 = params_distribution.copy()
params_distribution_0['soil'] = {'high_horizon': 0.05, 'low_horizon': 0.00, 'organic_above': 0.1,
                                 'mineral_above': 0.3, 'mineral_below': 0.40, 'organic_below': 0.00}
multiclass_dist = {0: params_distribution_0, 1: params_distribution}

def process_mintpy(
        sitename, year, pmintpy, pforcing, params_distribution, ecotype=False, xy_ref=None,
        overwrite=False, N=10000, Tf=None):

    ecotypename = 'singleensemble' if not ecotype else 'ecotype'
    pathout = paths['processed'] / f'{sitename}/{year}/{ecotypename}'
    fnunw = pmintpy / 'ph_history.tif'
    fnK = pmintpy / 'ph_history_cov.tif'
    fnmeta = pmintpy / 'meta.json'

    meta = read_meta_from_json(fnmeta)
    dailytemp, ind_scenes = forcing_merra_meta(pforcing, meta, year=year)

    if xy_ref is not None:
        data, geospatial = read_referenced_InSAR(
            fnunw, fnK, xy_ref, wavelength=meta['wavelength'], fndist=pathout / 'distance_cal.p',
            overwrite=True)
    else:
        raise NotImplementedError("xy_ref needed")

    if not ecotype:
        ec = None
    else:
        from scripts.ecotypes import reclassify
        fnlc = paths['ancillary'] / 'TNC/ecosystems_northern_alaska_jorgenson_2010.tif'
        fnec = pathout / 'ec.tif'
        ec = reclassify(fnlc, eclasses, geospatial, fnout=fnec, overwrite=overwrite)[0, ...]

    IP = InversionProcessorIS
    if not ecotype:
        IR = InversionResultsISMmap
    else:
        IR = MulticlassInversionResultsISMmap
    kwargs = {}

    predictor = StefanPredictor()

    def create_stratigraphy(dist):
        strat = StefanStratigraphySmoothingSpline(N=N, dist=dist)
        if Tf is not None:
            strat.ancillary['Tf'] = Tf
        return strat

    if ecotype:
        strats = {sc: create_stratigraphy(multiclass_dist[sc]) for sc in multiclass_dist}
        predens = MulticlassPredictionEnsemble(strats, predictor, geom=meta['geom'])
    else:
        strat = create_stratigraphy(params_distribution)
        predens = PredictionEnsemble(strat, predictor, geom=meta['geom'])
    data['ec'] = ec

    # store an atmospherically referenced deformation history; for internal appraisal of data quality
    geospatial.save_geotiff(data['s_obs'], pathout / 's_obs.tif')
    
    predens.predict(dailytemp)
    ip = IP(predens, geospatial=geospatial, blocksize=512, **kwargs)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], ec=data['ec'], pathout=pathout, n_jobs=-1, memory=False,
        overwrite=overwrite)
    ir.save(pathout / 'ir.p')
    ip.delete_temporary(pathout)
    ir = IR.from_file(pathout / 'ir.p')

    expecs = [('yf', 'mean'), ('e', 'mean'), ('e', 'var'), ]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(
            pathout, param=expec[0], etype=expec[1], hdf5=True, overwrite=overwrite, **kwargs)

if __name__ == '__main__':

    sitename = 'fort_yukon'
    year = 2023
    xy_ref = np.array([[576646.0, 7383520.0]]).T
    ecotype = False
    pmintpy = Path(f'/10TBstorage/Work/MintPy/{sitename}/s1/160/{year}')
    process_mintpy(
        sitename, year, pmintpy, pmintpy, params_distribution, ecotype=ecotype, xy_ref=xy_ref,
        overwrite=True)
