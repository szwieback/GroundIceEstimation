'''
Created on Oct 12, 2022

@author: simon
'''
import numpy as np
import os
import datetime

from scripts.happyvalley_analysis import read_results
from analysis import (read_K, add_atmospheric_K, read_referenced_motion)
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
        K = add_atmospheric_K(K, var_atmo)
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

def plot_retrieval(ax, res_site_year, lw_q=0.3, ymax=0.60, xlim=(-0.01, 0.80), c=None):
    from scripts.plotting import colslist
    from analysis import thaw_depth
    alpha = (res_site_year['frac_thawed']) ** 3
    ygrid = res_site_year['ygrid']
    e_mean = res_site_year['e_mean']
    e_q = res_site_year['e_quantile']
    td = thaw_depth(res_site_year['frac_thawed'], ygrid)
    print(td)
    ax.axhline(td, c='#cccccc', lw=0.2, alpha=0.5, zorder=1)
    if c is None: c = colslist[1]
    for jdepth in np.arange(ygrid.shape[0] - 1):
        ax.plot(
            e_mean[jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=1.0,
            c=c, alpha=alpha[jdepth])
        ax.plot(
            e_q[jdepth:jdepth + 2, 0], ygrid[jdepth:jdepth + 2], lw=lw_q,
            c=c, alpha=alpha[jdepth])
        ax.plot(
            e_q[jdepth:jdepth + 2, 1], ygrid[jdepth:jdepth + 2], lw=lw_q,
            c=c, alpha=alpha[jdepth])
    ax.fill_betweenx(
        ygrid, e_q[:, 0], e_q[:, 1], edgecolor='none', facecolor=c,
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
    fns_abs = os.path.join(paths['cores'], fns[site])
    y_grid_core = np.arange(150) / 100  # hard-coded for now

    e_grid = read_site(fns_abs)
    e_q_mean = bootstrap_percentiles(e_grid, (10, 90))
    e_mean = np.nanmean(e_grid, axis=0)
    ax.fill_betweenx(
        y_grid_core, e_q_mean[0,:], e_q_mean[1,:], edgecolor='none',
        facecolor=c, alpha=0.20)
    ax.plot(e_mean, y_grid_core, c=c, lw=1.2)

def plot_comparison(fnout=None, overwrite=False):
    from scripts.plotting import prepare_figure, colslist
    from string import ascii_lowercase
    sites = ('icecut', 'happyvalley')
    site_labels = ('Ice Cut', 'Happy Valley')
    years = (2022, 2019)
    res = {}
    for site in sites:
        res[site] = {}
        for year in years:
            res[site][year] = InSAR_results(site, year, overwrite=overwrite)

    yyticks = (0.0, 0.2, 0.4, 0.6)
    syticks = (0.0, 0.02, 0.04)
    fig, axs = prepare_figure(
        nrows=2, ncols=3, sharey=False, figsize=(1.40, 0.65), hspace=0.20, wspace=0.55,
        left=0.124, bottom=0.150, top=0.930, right=0.940)
    for jsite, site in enumerate(sites):
        axs[jsite, 2].axhline(0, lw=0.2, c='#cccccc')
        axs[jsite, 0].text(
            -0.50, 0.50, site_labels[jsite], ha='right', va='center',
            transform=axs[jsite, 0].transAxes, c='k', rotation=90)
        for jyear, year in enumerate(years):
            print(year, site)
            plot_retrieval(axs[jsite, jyear], res[site][year], c=colslist[2])
            plot_core(axs[jsite, jyear], site, c=colslist[0])
            plot_subsidence(axs[jsite, 2], res[site][year], c=colslist[jyear])
            axs[jsite, jyear].text(
                -0.29, 0.50, 'depth $y$ [cm]', ha='right', va='center',
                transform=axs[jsite, jyear].transAxes, rotation=90)
            axs[jsite, jyear].set_yticks(yyticks)
            axs[jsite, jyear].set_yticklabels((100 * np.array(yyticks)).astype(np.int16))
            axs[jsite, jyear].set_xticks((0.0, 0.3, 0.6))
        axs[jsite, 2].set_ylim((0.05, -0.02))
        axs[jsite, 2].set_yticks(syticks)
        axs[jsite, 2].set_yticklabels((100 * np.array(syticks)).astype(np.int16))
        axs[jsite, 2].text(
            -0.26, 0.50, '$s(t)$ [cm]', ha='right', va='center',
            transform=axs[jsite, 2].transAxes, rotation=90)
    xlab = ('excess ice $e$ [-]', 'excess ice $e$ [-]', 'date')
    coltitles = ('2022', '2019', 'subsidence')
    for jcol, lab in enumerate(xlab):
        axs[-1, jcol].text(
            0.50, -0.39, lab, ha='center', va='baseline', transform=axs[-1, jcol].transAxes)
        axs[0, jcol].text(
            0.50, 1.07, coltitles[jcol], ha='center', va='baseline', c='k',
            transform=axs[0, jcol].transAxes)
    for jax, ax in enumerate(axs.flatten()):
        xxlab = 0.98 if jax % 3 != 2 else 0.12
        ax.text(
            xxlab, 0.03, f'{ascii_lowercase[jax]})', ha='right', va='baseline',
            transform=ax.transAxes)
    xyearlab = 1.03
    axs[0, 2].text(
        xyearlab, 0.32, '2022', ha='left', transform=axs[0, 2].transAxes, c=colslist[0])
    axs[0, 2].text(
        xyearlab, 0.06, '2019', ha='left', transform=axs[0, 2].transAxes, c=colslist[1])
    axs[0, 0].text(
        0.30, 0.80, 'InSAR', ha='left', transform=axs[0, 0].transAxes, c=colslist[2])
    axs[0, 0].text(
        0.30, 0.34, 'cores', ha='left', transform=axs[0, 0].transAxes, c=colslist[0])
    axs[0, 0].text(
        0.80, 0.28, '$y_f$', ha='left', transform=axs[0, 0].transAxes, c='#cccccc')
    doy_ticks = (152, 182, 213, 244)
    axs[1, 2].set_xticks(doy_ticks)
    axs[1, 2].set_xticklabels(('Jun', 'Jul', 'Aug', 'Sep'))
    if fnout is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(fnout)

def comparison_thaw_depth(site='happyvalley', year=2022, ind=-1):
    from analysis import InversionResults, thaw_depth
    path_res = f'/home/simon/Work/gie/processed/Dalton_131_363/{site}/{year}/hadamard'

    ir = InversionResults.from_file(os.path.join(path_res, 'ir.p'))
    _rc_site = ir.geospatial.rowcol(xy_site[site])[:, 0]
    ir.lw = ir.lw[_rc_site[0], _rc_site[1],:][np.newaxis, ...]
    frac_thawed = ir.frac_thawed(ind_scene=ind)
    return thaw_depth(frac_thawed, ir.ygrid)[0]
    # print(ir.geospatial)
    # InSAR_results('icecut', 2022, overwrite=True)

if __name__ == '__main__':
    from scripts.pathnames import paths
    fnout = os.path.join(paths['figures'], f'northslope_comparison.pdf')
    # plot_comparison(fnout=fnout, overwrite=False)
    # raise
    from forcing import parse_dates
    # hv calm: 2019-08-12: 0.46
    dates = {
        'happyvalley': {2022: ('2022-06-06', '2022-08-17'), 2019: ('2019-05-18', '2019-09-06')},
        'icecut': {2022: ('2022-05-24', '2022-08-15'), 2019: ('2019-05-11', '2019-08-25')}}
    '2019-08-12'

    site = 'happyvalley'
    year = 2019
    d0, do = parse_dates(dates[site][year], strp='%Y-%m-%d')
    ind = (do - d0).days
    print(do)
    print(comparison_thaw_depth(site=site, year=year, ind=ind))
    
    # from forcing import read_daily_noaa_forcing
    # import pandas as pd
    # fnforcing = os.path.join(paths['forcing'], 'sagwon/sagwon.csv')
    # df = read_daily_noaa_forcing(fnforcing, convert_temperature=False)
    # year = 2019
    # d0 = {2022: '2022-05-24', 2021: '2021-05-25', 2019: '2019-05-11'}[year]
    # d1 = {2022: '2022-09-16', 2021: '2021-09-14', 2019: '2019-09-17'}[year]
    # dailytemp = (df.resample('D').mean())[pd.date_range(start=d0, end=d1)]
    # print(dailytemp)
    # # dailytemp[dailytemp < 0] = 0
    # print(np.sum(dailytemp))
