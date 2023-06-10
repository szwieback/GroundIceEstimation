'''
Created on Jun 8, 2023

@author: simon
'''
import numpy as np
import os
from analysis import (
    read_K, read_geotiff_geospatial, RationalQuadraticSepDiagCovMV, extract_reference, distance_to_ref, 
    add_nugget, Geospatial, save_object)

wvl = 0.055    

l = 12e3
var_rad = 1.25 ** 2 # radians squared
alpha = 0.9 # shape parameter
nugget_speckle = (0.2)**2 # radians squared

caldict = {'l': l, 'var_rad': var_rad, 'alpha': alpha, 'nugget_speckle': nugget_speckle}

def prepare_references(path0, reftype='regular'):
    fnunw = os.path.join(path0, 'unwrapped.geo.tif')
    geospatial = Geospatial.from_file(fnunw)
    print(geospatial)
    refs = {}
    refs_latlon = {}
    # regular references; geospatial x,y coords (or yx)
    refs['regular'] = np.array([   [1226, 357],
                                       [1099, 621],
                                       [883, 590],
                                       [859, 295],
                                       [824, 410],
                                       [722, 448],                   
                                       [674, 176],
                                       [527, 470],
                                       [343, 77], 
                                       [298, 319]])
    # additional points close to existing ref points for calibration
    refs['short'] = np.array([
                                   [775, 399],
                                   [555, 486],
                                   [565, 396],
                                   [364, 93], 
                                   [297, 302]])
    refs['all'] = np.concatenate((refs['regular'], refs['short']), axis=0)
    for _reftype in refs:
        rc_ref = refs[_reftype][...,::-1].T
        print(_reftype)
        refs_latlon[_reftype] = geospatial.xy(rc_ref)
        print(refs_latlon[_reftype])
    save_object(refs_latlon, os.path.join(path0, 'references_latlon.p'))
    return refs_latlon[reftype]

def evaluate_calibration(path0, caldict, overwrite=False):
    fnunw = os.path.join(path0, 'unwrapped.geo.tif')
    fnK = os.path.join(path0, 'K_vec.geo.tif')
    K, geospatial_K = read_K(fnK)
    unw, geospatial_unw = read_geotiff_geospatial(fnunw)
    assert geospatial_unw == geospatial_K    
    
    K = add_nugget(K, caldict['nugget_speckle'])
    ###
    # remove last scene due to unwrapping errors
    unw = unw[:-1, ...]
    K = K[:-1, :-1, ...]
    ###

    P = K.shape[0] + 1

    var_atmo = np.ones(P) * (caldict['var_rad'])  # in rad
    covmodel = RationalQuadraticSepDiagCovMV(caldict['l'], var_atmo, alpha=caldict['alpha'])

    xy_ref = prepare_references(path0, reftype='all')

    fndist = os.path.join(path0, 'distance_cal.p')
    dist = distance_to_ref(geospatial_unw, xy_ref, fndist=fndist, overwrite=overwrite)
    unw_ref, K_ref_comb, dist_matrix = extract_reference(K, unw, dist, geospatial_unw, xy_ref, covmodel)

    Nref = xy_ref.shape[1]
    dist_c = []
    svar_pred_c = [] #pred. semivariogram (single obs)
    svar_obs_c = [] #obs.semivariogram (single obs)
    for n1 in range(Nref):
        for n2 in range(n1+1, Nref):
            for p in range(P-1):
                # if 5 in (n1, n2): break
                dist_c.append(dist_matrix[n1, n2])
                ind1, ind2 = p + n1*(P-1), p + n2 * (P-1)
                svar_pred_c.append(
                    0.5 * (K_ref_comb[ind1, ind1] + K_ref_comb[ind2, ind2] - 2 * K_ref_comb[ind1, ind2]))
                svar_obs_c.append(0.5 * (unw_ref[ind1] - unw_ref[ind2]) ** 2)
    return dist_c, svar_pred_c, svar_obs_c

def plot_calibration(dist_c, svar_pred_c, svar_obs_c):
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.kernel_regression import KernelReg as kr 
    dist_plot = np.linspace(0, np.max(dist_c))
    def interpolate(svar, bw=2.5e3):
        _svar, _  = kr(
            endog=svar, exog=dist_c, var_type='c', bw=[bw], reg_type='ll').fit(data_predict=dist_plot)
        return _svar
    
    svar_obs_kr, svar_pred_kr = interpolate(svar_obs_c), interpolate(svar_pred_c)
    
    fig, ax = plt.subplots()
    alpha = 0.4
    alpha_l = 0.6
    ax.plot(dist_c, svar_obs_c, linestyle='none', marker='d', mfc='#999999', mec='none', ms=2, alpha=alpha)
    ax.plot(dist_c, svar_pred_c, linestyle='none', marker='o', mfc='k', mec='none', ms=2, alpha=alpha)
    ax.plot(dist_plot, svar_obs_kr, c='#999999', alpha=alpha_l)
    ax.plot(dist_plot, svar_pred_kr, c='k', alpha=alpha_l)
    ax.set_xlim((3e2, 1.5e4))
    ax.set_ylim((0, 2))
    plt.show()    

if __name__ == '__main__':
    path0 = f'/home/simon/Work/gie/processed/kivalina/2019'
    
    dist_c, svar_pred_c, svar_obs_c = evaluate_calibration(path0, caldict)
    plot_calibration(dist_c, svar_pred_c, svar_obs_c)
