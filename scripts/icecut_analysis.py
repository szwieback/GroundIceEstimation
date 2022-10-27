'''
Created on Oct 5, 2022

@author: simon
'''
import numpy as np
import os

from scripts.happyvalley_analysis import read_results

site = np.array((-148.8317, 69.0414))[:, np.newaxis]

def path_results(year):
    pathres = f'/home/simon/Work/gie/processed/Dalton_131_363/icecut/{year}/hadamard'
    return pathres

def invalid_mask(K, thresh, geospatial_K, geospatial, ind1=0, ind2=-1, wavelength=0.055):
    from scipy.ndimage import binary_dilation, binary_opening, binary_closing
    from analysis import add_atmospheric
    K = add_atmospheric(K, 0.0, wavelength=wavelength)
    K_last = K[ind1, ind1, ...] + K[ind2, ind2, ...] - 2 * K[ind1, ind2, ...]
    K_last_crop, _ = geospatial.warp(K_last, geospatial_K)
    s1 = np.array(
        [[ 0, 1, 0], [ 1, 1, 1], [0, 1, 0]])
    invalid = binary_closing(binary_opening(
        binary_dilation(K_last_crop > thresh ** 2, s1), s1, border_value=1), s1)
    # invalid = binary_dilation(binary_opening(binary_closing(K_last_crop > thresh ** 2, s1), s1), s1)
    return invalid

def icecut_map_profiles(fnout=None, overwrite=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from analysis import read_K
    from scripts.plotting import (
        cmap_e, colslist, _get_index, contrast, initialize_matplotlib,
        add_scalebar, plot_profile, add_arrow_line, ProfileInterpolator)
    years = (2022, 2019)
    fnimraw = '/home/simon/Work/gie/ancillary/Planet/20220620/20220620_211420_05_249d/analytic_sr_udm2/20220620_211420_05_249d_3B_AnalyticMS_SR.tif'
    fndemraw = '/home/simon/Work/gie/ancillary/ArcticDEM/46_18_10m_v3.0_reg_dem.tif'
    path0 = '/home/simon/Work/gie/processed/Dalton_131_363/'
    wavelength, thresh = 0.055, 4.3e-3
    upscale = 16

    cmap = cmap_e
    elim = (0.0, 0.5)
    xticks_im = (35, 70, 105, 140, 175)
    yticks_im = (31,)
    ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]

    profile = ((-148.7819, 69.0419), (-148.7560, 69.0408))  # (-148.7465, 69.0419))
    xy_ref = np.array([-148.7794, 69.0466])[:, np.newaxis]

    res0 = read_results(
        path_results(years[0]), fnimraw=fnimraw, fndemraw=fndemraw, upscale=upscale,
        overwrite=overwrite)
    res1 = read_results(path_results(years[1]), overwrite=overwrite)
    geospatial = res0['geospatial']
    assert res1['geospatial'] == geospatial

    # fig, axs = prepare_figure(ncols=3, nrows=3, sharex='none', sharey='none')
    fig = plt.figure()
    initialize_matplotlib()
    fig.set_size_inches((7.08, 2.37), forward=True)
    gs = gridspec.GridSpec(
        3, 14, left=0.007, right=0.995, top=0.950, bottom=0.120, wspace=4.10, hspace=0.30)
    axs = [[plt.subplot(gs[0, 0:4]), plt.subplot(gs[0, 4:8]), plt.subplot(gs[0, 8:12])],
           [plt.subplot(gs[1, 0:4]), plt.subplot(gs[1, 4:8]), plt.subplot(gs[1, 8:12])],
           [plt.subplot(gs[2, 0:4]), plt.subplot(gs[2, 4:9]), plt.subplot(gs[2, 9:])]]

    labels = [
        'a) 2022: excess ice 5--15 cm', 'b) 2022: excess ice 20--30 cm',
        'c) 2022: excess ice 40--50 cm', 'd) 2019: excess ice 5--15 cm',
        'e) 2019: excess ice 20--30 cm', 'f) 2019: excess ice 40--50 cm',
        'g) false-color image', 'h) 2022: transect T1', 'i) 2019: transect T1']

    for jyear, res in enumerate([res0, res1]):
        fnK = os.path.join(path0, str(years[jyear]), 'K_vec.geo.tif')
        K, geospatial_K = read_K(fnK)
        invalid = invalid_mask(
            K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)
        for jy, y in enumerate(ys):
            jy0, jy1 = _get_index(res['ygrid'], y[0]), _get_index(res['ygrid'], y[1])
            _e_mean = np.mean(res['e_mean'][..., jy0:jy1], axis=-1)
            _e_mean[invalid] = np.nan
            ax = axs[jyear][jy]
            im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
            ax.set_facecolor('#aaaaaa')
            ax.set_xticks(xticks_im)
            ax.set_yticks(yticks_im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(color='#dddddd', linewidth=0.4, alpha=0.5)
    optical = res0['optical'][::-1, ...][0:3]
    ax = axs[-1][0]
    ax.imshow(contrast(np.moveaxis(optical, 0, -1), percentiles=(2.0, 96.0)))
    ax.contour(
        res0['dem'][0, ...], colors=['#ffffff'], linewidths=0.4, alpha=0.7, levels=10)
    rc_ref = np.array(geospatial.upscaled(upscale).rowcol(xy_ref))[:, 0]
    ax.plot(
        rc_ref[1], rc_ref[0], linestyle='none', ms=2.5, marker='x',
        mec=colslist[0], mfc=colslist[0], zorder=9)
    ax.set_xticks(np.array(xticks_im) * upscale)
    ax.set_yticks(np.array(yticks_im) * upscale)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='#666666', linewidth=0.4)
    pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])
    rc = pi._rowcol_endpoints
    label = f'T1'
    add_arrow_line(
        ax, rc, label=label, c=colslist[0], lw=0.7, alpha=0.9, dlabel=(85, 210))
    _xy_site = geospatial.upscaled(upscale).rowcol(site)[:, 0]
    ax.plot(
        _xy_site[1], _xy_site[0], c=colslist[0], linestyle='none',
        marker='o', ms=5, mfc='none')
    ax.text(_xy_site[1] - 160, _xy_site[0] + 250, 'IC', c=colslist[0])
    add_scalebar(ax, geospatial.upscaled(upscale), length=1000, label='1 km', y=-0.2)

    xticks = [0, 250, 500, 750, 1000]
    yticks = (0.00, 0.25, 0.50)
    ymax = 0.50

    plabels = [
        (0.11, 'inactive fp'), (0.52, 'abandoned fp'), (0.93, 'slope')]
    x_ylabel = -0.11
    y_xlabel = -0.48
    plot_profile(
        axs[-1][1], res0['e_mean'], geospatial, profile, im_frac=res0['frac_thawed'],
        ymax=ymax, vlim=elim, ygrid=res0['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
        labels=None, x_ylabel=x_ylabel, y_xlabel=y_xlabel)
    axs[-1][1].text(
        0.05, 0.23, '$y_f$', c='#ffffff', transform=axs[-1][1].transAxes, alpha=0.6)
    plot_profile(
        axs[-1][2], res1['e_mean'], geospatial, profile, im_frac=res1['frac_thawed'],
        ymax=ymax, vlim=elim, ygrid=res1['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
        labels=plabels, x_ylabel=x_ylabel, y_xlabel=y_xlabel, y_plabels=0.83)

    cax = axs[0][-1].inset_axes([1.17, -0.75, 0.10, 1.20])
    cax.text(1.0, 1.18, '$e$ [-]', ha='center', va='baseline', transform=cax.transAxes)
    plt.colorbar(im_e, cax, shrink=0.5, orientation='vertical', ticks=[0.0, 0.25, 0.50])

    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.010, 1.060, lab, ha='left', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout)

if __name__ == '__main__':
    from scripts.pathnames import paths
    fnplot = os.path.join(paths['figures'], 'icecut.pdf')
    icecut_map_profiles(fnout=fnplot, overwrite=False)

'''year = '2019'#'2022'
pathres = f'/home/simon/Work/gie/processed/Dalton_131_363/icecut/{year}/hadamard'
fnimraw = '/home/simon/Work/gie/ancillary/Planet/20220620/20220620_211420_05_249d/analytic_sr_udm2/20220620_211420_05_249d_3B_AnalyticMS_SR.tif'
fnimres = os.path.join(pathres, 'optical.tif')
upscale = 8
site = np.array((-148.8317, 69.0414))[:, np.newaxis]

ir = InversionResults.from_file(os.path.join(pathres, 'ir.p'))
geospatial = ir.geospatial
ygrid = ir.ygrid
save_object(geospatial, os.path.join(pathres, 'geospatial.p'))
geospatial = load_object(os.path.join(pathres, 'geospatial.p'))
ygrid = np.arange(0, 1.5, step=2e-3)
e_mean = np.load(os.path.join(pathres, 'e_mean.npy'))
e_quantile = np.load(os.path.join(pathres, 'e_quantile.npy'))
frac_thawed = np.load(os.path.join(pathres, 'frac_thawed_None.npy'))
rc_site = geospatial.rowcol(site)
# rc_site[0,0] -=1
e_mean_site = e_mean[rc_site[0, 0], rc_site[1, 0],:]

e_quantile_site = e_quantile[rc_site[0, 0], rc_site[1, 0], ...]
frac_site = frac_thawed[rc_site[0, 0], rc_site[1, 0]]
print(ygrid[np.nonzero(frac_site < 1 / 2)[0][0]])

import matplotlib.pyplot as plt
from scripts.plotting import prepare_figure, cmap_e, colslist, _get_index, contrast
fig, ax = prepare_figure(nrows=1, ncols=1)
ax.fill_betweenx(ygrid, e_quantile_site[:, 0], e_quantile_site[:, 1], edgecolor='none', facecolor=colslist[0], alpha=0.07)
ax.plot(e_mean_site, ygrid, c=colslist[0])
ax.set_ylim((0.60, 0))
plt.show()

cmap = cmap_e
elim = (0.0, 0.5)
xticks_im = (25, 65, 105, 145)
yticks_im = (10, 50)
ys = [(0.05, 0.15), (0.20, 0.30), (0.50, 0.60)]
fig, axs = prepare_figure(ncols=3, nrows=2, sharex='none', sharey='none')

optical = resample_dem(geospatial, fnimraw, fnimres, upscale=upscale)

for jy, y in enumerate(ys):
    _e_mean = np.mean(
        e_mean[..., _get_index(ygrid, y[0]):_get_index(ygrid, y[1])], axis=-1)
    # _e_mean[invalid] = np.nan
    ax = axs[0, jy]
    im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
    ax.set_facecolor('#aaaaaa')
    ax.set_xticks(xticks_im)
    ax.set_yticks(yticks_im)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='#dddddd', linewidth=0.4)
optical = optical[::-1, ...][0:3]
ax = axs[1][0]
ax.imshow(contrast(np.moveaxis(optical, 0, -1)))
plt.show()
'''
