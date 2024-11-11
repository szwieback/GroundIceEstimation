'''
Created on Oct 22, 2024

@author: simon
'''
import numpy as np
from pathlib import Path

from scripts.happyvalley_analysis import read_results
from analysis import InversionResultsIS
from scripts.pathnames import paths
from scripts.pokerflats import geom, datesstr, read_InSAR, xy_ref, wavelength, fns_unw_offset, var_atmo, InSAR_site
thresh = 4.3e-3

p0 = paths['processed'] / 'pokerflats'

sites = {'tower': (-147.4874, 65.1239), 'clearing': (-147.47948, 65.12625)}


def path_results(year, method='hadamard'):
    pathres = p0 / f'{year}/{method}'
    return pathres

def plot_map_subsidence(year, fnout=None):
    import matplotlib.pyplot as plt
    import datetime
    from scripts.plotting import (
        prepare_figure, cmap_e, cmap_s, _get_index, add_scalebar)
    pstack = paths['stacks'] / 'Fairbanks_131_373'/ f'{year}'

    dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    doys = [d.timetuple().tm_yday for d in dates]
    s_obs, K, geospatial = read_InSAR(
        pstack, wavelength, xy_ref=xy_ref, fns_unw_offset=fns_unw_offset.get(year, []), var_atmo=var_atmo)

    cmap = cmap_e
    elim = (0.0, 0.5)
    slim = (-0.1, 0.1)  # (-0.06, 0.06)
    xticks_im = (35, 70, 105, 140, 175)
    yticks_im = (31,)
    ys = (0.40, 0.50)

    res0 = read_results(path_results(year, method=method), overwrite=False)
    assert geospatial == res0['geospatial']
    fig, axs = prepare_figure(
        ncols=2, nrows=1, figsize=(1.70, 0.52), sharex=True, sharey=True, left=0.04, right=0.98,
        bottom=0.18, top=0.95, hspace=0.10, wspace=0.10, remove_spines=False)
    inds = [(7, -1), (2, 7)]
    # labels = [
    #     'a) excess ice 40--50 cm', 'b) $s$ Jul 28 -- Sep 14, 2022',
    #     'c) $s$ Jun 10 -- Sep 14, 2022', 'd) excess ice 40--50 cm',
    #     'e) $s$ Aug 01 -- Sep 06, 2019', 'f) $s$ Jun 02 -- Sep 06, 2019',]

    # invalid = invalid_mask(
    #     K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)
    #
    # jy0, jy1 = _get_index(resl[jyear]['ygrid'], ys[0]), _get_index(resl[jyear]['ygrid'], ys[1])
    # _e_mean = np.mean(resl[jyear]['e_mean'][..., jy0:jy1], axis=-1)
    # _e_mean[invalid] = np.nan
    # ax = axs[jyear, 0]
    # im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
    s_obs_v = s_obs / np.cos(geom['ia'])
    s_obs_v = np.concatenate((np.zeros((1,) + s_obs_v.shape[1:]), s_obs_v), axis=0)
    for jind, ind in enumerate(inds):
        s_obs_diff = s_obs_v[ind[1], ...] - s_obs_v[ind[0], ...]
        print(dates[ind[0]], dates[ind[1]])
        # s_obs_diff_crop, _ = geospatial.warp(s_obs_diff, sresl[jyear]['geospatial'])
        s_obs_diff_crop = s_obs_diff
        # s_obs_diff_crop[invalid] = np.nan
        ax = axs[jind]
        im_s = ax.imshow(s_obs_diff_crop, cmap=cmap_s, vmin=slim[0], vmax=slim[1])

    for ax in axs.flatten():
        ax.set_facecolor('#aaaaaa')
        ax.set_xticks(xticks_im)
        ax.set_yticks(yticks_im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='#dddddd', linewidth=0.4, alpha=0.5)
    # unwrapping errors early on; need to zoom in on smaller areas
    # apparent uplift over decid. broadleaf
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout, dpi=450)

def plot_site(sitename, year, method='hadamard', fnout=None):
    import datetime
    import matplotlib.dates as mdates
    from scripts.plotting import prepare_figure, colslist
    import matplotlib.pyplot as plt
    
    res0 = read_results(path_results(year, method=method), overwrite=False)
    geospatial = res0['geospatial']
    xy_site = sites[sitename]
    rc = geospatial.rowcol(xy_site)
    fig, axs = prepare_figure(ncols=2, sharex='none', sharey='none', wspace=0.4, left=0.11, right=0.98)
    ygrid = res0['ygrid']
    e_mean = res0['e_mean'][rc[0], rc[1], ...]
    e_q = res0['e_quantile'][rc[0], rc[1], ...]
    frac_thawed = res0['frac_thawed'][rc[0], rc[1], ...]
    ax = axs[0]
    c, lw_q = colslist[0], 0.5
    alpha = (frac_thawed) ** 3
    alpha[alpha > 1.0] = 1.0
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
    axs[0].set_ylim((0.9, 0.0))
    
    ax = axs[1]
    pstack = paths['stacks'] / 'Fairbanks_131_373'/ f'{year}'        
    s_obs, K, geospatial = read_InSAR(
        pstack, wavelength, xy_ref=xy_ref, fns_unw_offset=fns_unw_offset.get(year, []), var_atmo=var_atmo)
    dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    s_obs_site, sd_site = InSAR_site(
        s_obs, K, geospatial, xy_site, geom, vertical=True, vertical_sign_flip=True)
    ax.errorbar(dates, s_obs_site, yerr=sd_site, color='none', ecolor=c, alpha=0.5, lw=0.5)
    ax.plot(dates, s_obs_site, c=c) 
    ax.axhline(0, c='#eeeeee', lw=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_ylim((-0.06, 0.01))    
    labels = (['excess ice $e$ [-]', 'deformation $s$ [m]'])
    for lab, ax in zip(labels, axs):
        ax.text(-0.28, 0.50, lab, ha='left', va='center', rotation=90, transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)
    

# plot profile; then repeat with talik and with late-season index (for both)

# map of estimated ice content

if __name__ == '__main__':
    year = 2024
    method = 'hadamard'

    # plot_map_subsidence(2024)
    
    sitename = 'clearing'
    plot_site(sitename, year, method)
