'''
Created on Oct 12, 2022

@author: simon
'''
import numpy as np
import os
import datetime

from scripts.happyvalley_analysis import read_results
from analysis import (read_K, add_atmospheric, read_referenced_motion)
from ioput import save_object, load_object

fngpkg = '/home/simon/Work/gie/processed/Dalton_131_363/2019_unw_offset.gpkg'

xy_site = {'icecut': np.array((-148.8317, 69.0414))[:, np.newaxis],
           'happyvalley': np.array((-148.8437, 69.1548))[:, np.newaxis]}
geom = {'icecut': {'ia': 43.54 / 180 * np.pi}, 'happyvalley': {'ia': 38.40 / 180 * np.pi}}
wavelength = 0.055
var_atmo = (4e-3) ** 2
xy_ref = {'icecut': np.array([-148.7794, 69.0466])[:, np.newaxis],
          'happyvalley': np.array([-148.8063, 69.1616])[:, np.newaxis]}
fns_unw_offset = {
    'icecut': {2019: [], 2022: []},
    'happyvalley': {2019: [(7, fngpkg)], 2022: []}}
datesstr = {
    'icecut': {
        2022: ('20220529', '20220610', '20220622', '20220704', '20220716', '20220728',
               '20220809', '20220821', '20220902', '20220914'),
        2019: ('20190521', '20190602', '20190614', '20190626', '20190708', '20190720',
               '20190801', '20190813', '20190825', '20190906')},
    'happyvalley': {
        2022: ('20220610', '20220622', '20220704', '20220716', '20220728',
               '20220809', '20220821', '20220902', '20220914'),
        2019: ('20190602', '20190614', '20190626', '20190708', '20190720',
               '20190801', '20190813', '20190825', '20190906')}}

def adjust_covariance(C_obs, ind=-3):
    C_obs_b = np.zeros((C_obs.shape[0] + 1,) * 2)
    C_obs_b[1:, 1:] = C_obs
    A = np.eye(C_obs_b.shape[0])
    A[:, ind] -= 1
    return np.linalg.multi_dot((A, C_obs_b, A.T))

def InSAR_results(site, year, rmethod='hadamard', overwrite=False):
    pathres = f'/home/simon/Work/gie/processed/Dalton_131_363/{site}/{year}/{rmethod}'
    path0 = f'/home/simon/Work/gie/processed/Dalton_131_363/{year}'

    fnsite = os.path.join(pathres, 'site.p')
    if not os.path.exists(fnsite) or overwrite:
        res = read_results(pathres)
        fnunw = os.path.join(path0, 'unwrapped.geo.tif')
        fnK = os.path.join(path0, 'K_vec.geo.tif')
        dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[site][year]]
        doys = [d.timetuple().tm_yday for d in dates]
        K, geospatial_K = read_K(fnK)
        K = add_atmospheric(K, var_atmo)
        s_obs, geospatial = read_referenced_motion(
            fnunw, xy=xy_ref[site], wavelength=wavelength,
            fns_unw_offset=fns_unw_offset[site][year])
        assert geospatial == geospatial_K

        if site == 'happyvalley':  # remove first acq because still a lot of snow
            K = K[1:, 1:, ...]
            s_obs = s_obs[1:, ...] - s_obs[0, ...][np.newaxis, ...]

        _rc_site = geospatial.rowcol(xy_site[site])[:, 0]
        C_obs = K[..., _rc_site[0], _rc_site[1]]
        s_obs = s_obs[..., _rc_site[0], _rc_site[1]]
        s_obs_v = s_obs / np.cos(geom[site]['ia'])
        C_obs_v = C_obs / (np.cos(geom[site]['ia']) ** 2)
        # C_obs_v_last = adjust_covariance(C_obs_v, ind=-1)
        _rc_site = res['geospatial'].rowcol(xy_site[site])[:, 0]
        e_mean = res['e_mean'][_rc_site[0], _rc_site[1],:]
        e_quantile = res['e_quantile'][ _rc_site[0], _rc_site[1], ...]
        frac_thawed = res['frac_thawed'][_rc_site[0], _rc_site[1],:]
        siteres = {
            'C_obs_v': C_obs_v, 's_obs_v': s_obs_v, 'e_mean': e_mean,
            'e_quantile': e_quantile, 'frac_thawed': frac_thawed, 'ygrid': res['ygrid'],
            'dates': dates, 'doys': doys}
        save_object(siteres, fnsite)
    else:
        siteres = load_object(fnsite)
    return siteres

def plot_retrieval(ax, res_site_year, lw_q=0.3, ymax=0.65, xlim=(-0.01, 0.80)):
    from scripts.plotting import colslist
    alpha = (res_site_year['frac_thawed']) ** 2
    ygrid = res_site_year['ygrid']
    e_mean = res_site_year['e_mean']
    e_q = res_site_year['e_quantile']

    for jdepth in np.arange(ygrid.shape[0] - 1):
        ax.plot(
            e_mean[jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=1.0,
            c=colslist[0], alpha=alpha[jdepth])
        ax.plot(
            e_q[jdepth:jdepth + 2, 0], ygrid[jdepth:jdepth + 2], lw=lw_q,
            c=colslist[0], alpha=alpha[jdepth])
        ax.plot(
            e_q[jdepth:jdepth + 2, 1], ygrid[jdepth:jdepth + 2], lw=lw_q,
            c=colslist[0], alpha=alpha[jdepth])
    ax.fill_betweenx(
        ygrid, e_q[:, 0], e_q[:, 1], edgecolor='none', facecolor=colslist[1],
        alpha=0.10)
    ax.set_ylim((ymax, ygrid[0]))
    ax.set_xlim(xlim)

def plot_subsidence(ax, res_site_year, c='#000000'):
    doys = res_site_year['doys']
    s_obs_v = res_site_year['s_obs_v']
    std_obs_v = np.sqrt(np.diag(res_site_year['C_obs_v']))
    sm, sp = s_obs_v - std_obs_v, s_obs_v + std_obs_v
    s_obs_v_0 = np.concatenate(([0], s_obs_v))
    ax.plot(doys, s_obs_v_0, c=c, lw=1.0, alpha=0.5)
    for jdoy, doy in enumerate(doys[1:]):
        ax.plot((doy, doy), (sm[jdoy], sp[jdoy]), lw=0.5, alpha=0.8, c=c)
    ax.plot(doys, s_obs_v_0, c=c, linestyle='none', marker='o', mec=c, mfc='w', ms=2)

def plot_core(ax, site, c='#000000'):
    from scripts.pathnames import paths
    from scripts.core_analysis import read_site, bootstrap_percentiles
    fns = {
        'icecut': 'FSA_Dalton_IC_2022_20220826.xlsx',
        'happyvalley': 'FSA_Dalton_HV_2022_20220826.xlsx'}
    fns_abs = {site: os.path.join(paths['cores'], fns[site]) for site in fns}
    y_grid_core = np.arange(150) / 100  # hard-coded for now


    e_grid = read_site(fns_abs[site])
    e_q_mean = bootstrap_percentiles(e_grid, (10, 90))
    e_mean = np.nanmean(e_grid, axis=0)
    print(np.mean(e_mean[5:25]) * 0.2)
    ax.fill_betweenx(
        y_grid_core, e_q_mean[0,:], e_q_mean[1,:], edgecolor='none',
        facecolor=c, alpha=0.20)
    ax.plot(e_mean, y_grid_core, c=c, lw=1.2)

def plot_comparison(overwrite=False):
    from scripts.plotting import prepare_figure, colslist
    import matplotlib.pyplot as plt
    sites = ('icecut', 'happyvalley')
    years = (2022, 2019)
    res = {}
    for site in sites:
        res[site] = {}
        for year in years:
            res[site][year] = InSAR_results(site, year, overwrite=overwrite)

    fig, axs = prepare_figure(nrows=2, ncols=3, sharey=False)
    for jsite, site in enumerate(sites):
        axs[jsite, 2].axhline(0, lw=0.2, c='#cccccc')
        for jyear, year in enumerate(years):
            plot_core(axs[jsite, jyear], site, c=colslist[2])
            plot_retrieval(axs[jsite, jyear], res[site][year])
            plot_subsidence(axs[jsite, 2], res[site][year], c=colslist[jyear])
        axs[jsite, 2].set_ylim((0.05, -0.02))

    

    plt.show()

if __name__ == '__main__':
    plot_comparison(overwrite=False)

