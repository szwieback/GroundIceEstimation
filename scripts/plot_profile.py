'''
Created on Oct 16, 2024

@author: simon
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from analysis import load_object, save_object, InversionResultsIS, read_K, MulticlassInversionResultsIS
from scripts.kivalina_analysis import resample_dem
from scripts.plotting import prepare_figure

def read_InSAR(path0, wavelength, var_atmo=None, fns_unw_offset=(), xy_ref=None, fill_nan=False):
    from analysis import (read_K, add_atmospheric_K, read_referenced_motion)
    fnunw = path0 / 'unwrapped.geo.tif'
    fnK = path0 / 'K_vec.geo.tif'
    K, geospatial_K = read_K(fnK)
    s_obs, geospatial = read_referenced_motion(
        fnunw, xy=xy_ref, wavelength=wavelength, fns_unw_offset=fns_unw_offset)

    K = add_atmospheric_K(K, var_atmo)
    assert geospatial == geospatial_K
    if fill_nan:
        s_obs[np.isnan(s_obs)] = 0.0
        K[np.isnan(K)] = 1.0
    return s_obs, K, geospatial

def InSAR_site(s_obs, K, geospatial, xy_site, geom=None, vertical=False, vertical_sign_flip=True):
    rc = geospatial.rowcol(xy_site)

    s_obs_site = np.concatenate((np.zeros((1,) + s_obs.shape[1:]), s_obs), axis=0)[:, rc[0], rc[1]]
    sd_site = np.concatenate([(0, ), np.sqrt(np.diag(K[..., rc[0], rc[1]]))])
                             
    if vertical:
        assert geom is not None
        s_obs_site = s_obs_site / np.cos(geom['ia'])
        sd_site = sd_site / np.cos(geom['ia'])
        if vertical_sign_flip:
            s_obs_site *= -1# make positive up instead of down
    return s_obs_site, sd_site

def read_results(pathres, IR=None, fnimraw=None, fndemraw=None, upscale=8, overwrite=True):
    fngeospatial = pathres / 'geospatial.p'
    fnygrid = pathres / 'ygrid.p'
    if not fngeospatial.exists() or not fnygrid.exists() or overwrite:
        if IR is None: IR = InversionResultsIS
        ir = IR.from_file(pathres / 'ir.p')
        geospatial = ir.geospatial
        ygrid = ir.ygrid
        save_object(geospatial, fngeospatial)
        save_object(ygrid, fnygrid)
    else:
        geospatial = load_object(fngeospatial)
        ygrid = load_object(fnygrid)
    res = {'ygrid': ygrid, 'geospatial': geospatial}
    res['e_mean'] = np.load(pathres / 'e_mean.npy')
    try:
        res['e_quantile'] = np.load(pathres / 'e_quantile.npy')
    except:
        pass
    res['frac_thawed'] = np.load(pathres / 'frac_thawed_None.npy')
    try:
        res['e_mean_period_mean'] = np.load(pathres / 'e_mean_period_mean.npy')
        res['e_mean_period_quantile'] = np.load(pathres / 'e_mean_period_quantile.npy')
    except:
        pass
    if fnimraw is not None:
        fnimres = pathres / 'optical.tif'
        res['optical'] = resample_dem(
            geospatial, fnimraw, fnimres, upscale=upscale, overwrite=overwrite)
    if fndemraw is not None:
        fndemres = pathres / 'dem.tif'
        res['dem'] = resample_dem(
            geospatial, fndemraw, fndemres, upscale=upscale, overwrite=overwrite)
    return res

def plot_profile(path_res, xy_site, fnout=None):
    res = read_results(
            path_res, fnimraw=None, fndemraw=None, upscale=None, overwrite=False)
    _rc_site = res['geospatial'].rowcol(xy_site)[:, 0]
    e_mean = res['e_mean'][_rc_site[0], _rc_site[1],:]
    e_q = res['e_quantile'][ _rc_site[0], _rc_site[1], ...]
    frac_thawed = res['frac_thawed'][_rc_site[0], _rc_site[1],:]

    fig, ax = prepare_figure(1, 1, figsize=(0.8, 0.6), left=0.17, bottom=0.17)
    ygrid = res['ygrid']

    c = '#333333'
    ymax = 0.7

    alpha = (frac_thawed) ** 3
    alpha[alpha > 1.0] = 1.0

    for jdepth in np.arange(ygrid.shape[0] - 1):
        ax.plot(
            e_mean[jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=1.0,
            c=c, alpha=alpha[jdepth])
        ax.plot(
            e_q[jdepth:jdepth + 2, 0], ygrid[jdepth:jdepth + 2], lw=0.1,
            c=c, alpha=alpha[jdepth])
        ax.plot(
            e_q[jdepth:jdepth + 2, 1], ygrid[jdepth:jdepth + 2], lw=0.1,
            c=c, alpha=alpha[jdepth])
    ax.fill_betweenx(
        ygrid, e_q[:, 0], e_q[:, 1], edgecolor='none', facecolor=c,
        alpha=0.04)

    ax.set_ylim((ymax, ygrid[0]))
    ax.set_xlim((0.0, 0.5))
    ax.text(0.50, -0.15, 'excess ice $e$ [-]', transform=ax.transAxes, ha='center', va='top')
    ax.text(-0.15, 0.50, 'depth $y$ [m]', transform=ax.transAxes, rotation=90, ha='right', va='center')
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

if __name__ == '__main__':
    year, method = 2024, 'hadamard'
    wavelength, thresh = 0.055, 4.3e-3

    from scripts.pathnames import paths
    print(paths.keys())
    path_res = paths['processed'] / f'Dalton_131_363/happyvalley/{year}/{method}/'
    xy_site = np.array((-148.8437, 69.1548))[:, np.newaxis]

    plot_profile(path_res, xy_site, fnout=paths['figures'] / 'happyvalley_2024.pdf')


    path_res = paths['processed'] / f'Dalton_131_363/icecut/{year}/{method}/'
    xy_site = np.array((-148.8317, 69.0414))[:, np.newaxis]

    plot_profile(path_res, xy_site, fnout=paths['figures'] / 'icecut_2024.pdf')

