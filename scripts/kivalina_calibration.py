'''
Created on Jun 8, 2023

@author: simon
'''
import numpy as np
from pathlib import Path

from analysis import (
    read_K, read_geotiff_geospatial, RationalQuadraticSepDiagCovMV, extract_reference, distance_to_ref,
    add_nugget, Geospatial, save_object)

wvl = 0.055

l = 12e3
var_rad = 1.25 ** 2  # radians squared
alpha = 0.9  # shape parameter
nugget_speckle = (0.2) ** 2  # radians squared

caldict = {'l': l, 'var_rad': var_rad, 'alpha': alpha, 'nugget_speckle': nugget_speckle}

def references(year=2019):
    refs = {}
    refs['regular'] = np.array([    [-164.25190, 67.83774],
                                    [-164.31991, 67.87082],
                                    [-164.39611, 67.81081],
                                    [-164.52571, 67.81786],
                                    [-164.54011, 67.88491],
                                    [-164.56111, 67.85877],
                                    [-164.62231, 67.85014],
                                    [-164.65111, 67.91196],
                                    [-164.73931, 67.84514],
                                    [-164.84971, 67.93446],
                                    [-164.87671, 67.87946]])
    if year == 2018:
        refs['regular'][6,:] = [-164.62411, 67.85179]

    refs['short'] = np.array([  [-164.59051, 67.86127],
                                [-164.72251, 67.84150],
                                [-164.71651, 67.86195],
                                [-164.83711, 67.93082],
                                [-164.87731, 67.88332]])
    refs['all'] = np.concatenate((refs['regular'], refs['short']), axis=0)
    return refs

def prepare_references(path0, refs, reftype='regular'):
    fnunw = path0 / 'unwrapped.geo.tif'
    geospatial = Geospatial.from_file(fnunw)
    refs_latlon = {_reftype: refs[_reftype][::-1, ...].T for _reftype in refs}
    save_object(refs_latlon, path0 / 'references_latlon.p')
    np.savetxt(
        path0 / 'references_latlon.csv', refs_latlon['all'].T, delimiter=',', fmt='%10.5f')
    return refs_latlon[reftype]

def evaluate_calibration(path0, caldict, overwrite=False):
    fnunw = path0 / 'unwrapped_corr.geo.tif'
    fnK = path0 / 'K_vec.geo.tif'
    K, geospatial_K = read_K(fnK)
    unw, geospatial_unw = read_geotiff_geospatial(fnunw)
    assert geospatial_unw == geospatial_K

    K = add_nugget(K, caldict['nugget_speckle'])
    # ##
    # remove last scene due to unwrapping errors
    unw = unw[:-1, ...]
    K = K[:-1,:-1, ...]
    # ##

    P = K.shape[0] + 1

    var_atmo = np.ones(P) * (caldict['var_rad'])  # in rad
    covmodel = RationalQuadraticSepDiagCovMV(caldict['l'], var_atmo, alpha=caldict['alpha'])

    refs = references()
    xy_ref = prepare_references(path0, refs, reftype='all')

    fndist = path0 / 'distance_cal.p'
    dist = distance_to_ref(geospatial_unw, xy_ref, fndist=fndist, overwrite=overwrite)
    unw_ref, K_ref_comb, dist_matrix = extract_reference(K, unw, dist, geospatial_unw, xy_ref, covmodel)

    Nref = xy_ref.shape[1]
    dist_c = []
    svar_pred_c = []  # pred. semivariogram (single obs)
    svar_obs_c = []  # obs.semivariogram (single obs)
    for n1 in range(Nref):
        for n2 in range(n1 + 1, Nref):
            for p in range(P - 1):
                # if 5 in (n1, n2): break
                dist_c.append(dist_matrix[n1, n2])
                ind1, ind2 = p + n1 * (P - 1), p + n2 * (P - 1)
                svar_pred_c.append(
                    0.5 * (K_ref_comb[ind1, ind1] + K_ref_comb[ind2, ind2] - 2 * K_ref_comb[ind1, ind2]))
                svar_obs_c.append(0.5 * (unw_ref[ind1] - unw_ref[ind2]) ** 2)
    return dist_c, svar_pred_c, svar_obs_c

def plot_calibration(dist_c, svar_pred_c, svar_obs_c, fnout=None):
    from scripts.plotting import prepare_figure, colslist
    from statsmodels.nonparametric.kernel_regression import KernelReg as kr
    dist_plot = np.linspace(0, np.max(dist_c))
    def interpolate(svar, bw=2e3):
        _svar, _ = kr(
            endog=svar, exog=dist_c, var_type='c', bw=[bw], reg_type='ll').fit(data_predict=dist_plot)
        return np.array(_svar)

    svar_obs_kr, svar_pred_kr = interpolate(svar_obs_c), interpolate(svar_pred_c)
    conv = 1e6 * (wvl / (4 * np.pi)) ** 2

    fig, ax = prepare_figure(nrows=1, ncols=1, figsize=(0.96, 0.6), bottom=0.17, left=0.13, top=0.98)
    alpha = 0.4
    alpha_l = 1.0
    ho1 = ax.plot(
        dist_c, np.array(svar_obs_c) * conv, linestyle='none', marker='d', mfc=colslist[3], mec='none',
        ms=1.5, alpha=alpha)
    ho0 = ax.plot(
        dist_plot, svar_obs_kr * conv, c=colslist[3], alpha=alpha_l)
    hp = ax.plot(
        dist_c, np.array(svar_pred_c) * conv, linestyle='none', mfc=colslist[0], marker='o', mec='none',
        ms=2.0)
    # ax.plot(
    #     dist_plot, svar_pred_kr * conv, c='k', alpha=alpha_l)
    ax.set_xlim((3e2, 1.4e4))
    xticks = np.array((2500, 5000, 7500, 10000, 12500), dtype=np.int64)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks * 1e-3)
    ax.set_ylim((0, 68))
    ax.spines['left'].set_bounds(0, 50)
    ax.set_yticks((0, 25, 50))
    ax.text(0.50, -0.19, 'distance [km]', ha='center', va='baseline', transform=ax.transAxes)
    ylabel = '$\\hat{\\mathbf{d}}_{\\mathrm{r}}$ semi-variogram [mm$^{2}$]'
    ax.text(-0.105, 0.350, ylabel, transform=ax.transAxes, rotation=90, va='center', ha='right')
    ax.legend(
        handles=(ho0[0], ho1[0], hp[0]), labels=('observed (smoothed)', 'observed (raw)', 'modeled'),
        loc='upper left', bbox_to_anchor=(-0.16, 1.06), frameon=False, markerscale=1.2, borderpad=0.2,
        labelspacing=0.2, handletextpad=0.7, handlelength=1.0)
    if fnout is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(fnout)

if __name__ == '__main__':
    path0 = Path('/home/simon/Work/gie/')
    fnout = path0 / 'figures' /'kivalina_calibration.pdf'
    path1 = path0 / 'processed' / 'kivalina', '2019_index'
    dist_c, svar_pred_c, svar_obs_c = evaluate_calibration(path1, caldict, overwrite=False)
    plot_calibration(dist_c, svar_pred_c, svar_obs_c, fnout=fnout)

