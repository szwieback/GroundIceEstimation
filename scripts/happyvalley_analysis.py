'''
Created on Oct 5, 2022

@author: simon
'''
import numpy as np
import os
from analysis import Geospatial, save_geotiff, load_object, save_object, InversionResults
from scripts.kivalina_analysis import resample_dem

site = np.array((-148.8437, 69.1548))[:, np.newaxis]

def read_results(year, fnimraw=None, fndemraw=None, upscale=8):
    pathres = f'/home/simon/Work/gie/processed/Dalton_131_363/happyvalley/{year}/hadamard'
    # ir = InversionResults.from_file(os.path.join(pathres, 'ir.p'))
    # geospatial = ir.geospatial
    # ygrid = ir.ygrid
    # save_object(geospatial, os.path.join(pathres, 'geospatial.p'))
    geospatial = load_object(os.path.join(pathres, 'geospatial.p'))
    ygrid = np.arange(0, 1.5, step=2e-3)
    res = {'ygrid': ygrid, 'geospatial': geospatial}
    res['e_mean'] = np.load(os.path.join(pathres, 'e_mean.npy'))
    res['e_quantile'] = np.load(os.path.join(pathres, 'e_quantile.npy'))
    res['frac_thawed'] = np.load(os.path.join(pathres, 'frac_thawed_None.npy'))
    if fnimraw is not None:
        fnimres = os.path.join(pathres, 'optical.tif')
        res['optical'] = resample_dem(geospatial, fnimraw, fnimres, upscale=upscale)
    if fndemraw is not None:
        fndemres = os.path.join(pathres, 'dem.tif')
        res['dem'] = resample_dem(geospatial, fndemraw, fndemres, upscale=upscale)
    return res

def site_analysis():
    pass
    # rc_site = geospatial.rowcol(site)
    # e_mean_site = e_mean[rc_site[0, 0], rc_site[1, 0],:]
    # e_quantile_site = e_quantile[rc_site[0, 0], rc_site[1, 0], ...]
    # frac_site = frac_thawed[rc_site[0, 0], rc_site[1, 0]]
    # print(ygrid[np.nonzero(frac_site < 1 / 2)[0][0]])
    # fig, ax = prepare_figure(nrows=1, ncols=1)
    # ax.fill_betweenx(ygrid, e_quantile_site[:, 0], e_quantile_site[:, 1], edgecolor='none', facecolor=colslist[0], alpha=0.07)
    # ax.plot(e_mean_site, ygrid, c=colslist[0])
    # ax.set_ylim((0.55, 0))
    # plt.show()

def happyvalley_map_profiles(fnout=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scripts.plotting import (
        prepare_figure, cmap_e, colslist, _get_index, contrast, initialize_matplotlib,
        add_scalebar, plot_profile, add_arrow_line, ProfileInterpolator)
    years = (2022, 2019)
    fnimraw = '/home/simon/Work/gie/ancillary/Planet/20220620/20220620_211417_74_249d/analytic_sr_udm2/20220620_211417_74_249d_3B_AnalyticMS_SR.tif'
    fndemraw = '/home/simon/Work/gie/ancillary/ArcticDEM/46_18_10m_v3.0_reg_dem.tif'
    upscale = 8

    cmap = cmap_e
    elim = (0.0, 0.5)
    xticks_im = (25, 65, 105, 145)
    yticks_im = (25, 65, 105)
    ys = [(0.05, 0.15), (0.20, 0.30), (0.45, 0.55)]

    # profile = ((-148.8013, 69.1609), (-148.7717, 69.1609))
    profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))

    res0 = read_results(years[0], fnimraw=fnimraw, fndemraw=fndemraw, upscale=upscale)
    res1 = read_results(years[1])
    geospatial = res0['geospatial']
    assert res1['geospatial'] == geospatial

    # fig, axs = prepare_figure(ncols=3, nrows=3, sharex='none', sharey='none')
    fig = plt.figure()
    initialize_matplotlib()
    fig.set_size_inches((6.00, 3.85), forward=True)
    gs = gridspec.GridSpec(
        3, 14, left=0.007, right=0.995, top=0.968, bottom=0.078, wspace=6.10, hspace=0.18)
    axs = [[plt.subplot(gs[0, 0:4]), plt.subplot(gs[0, 4:8]), plt.subplot(gs[0, 8:12])],
           [plt.subplot(gs[1, 0:4]), plt.subplot(gs[1, 4:8]), plt.subplot(gs[1, 8:12])],
           [plt.subplot(gs[2, 0:4]), plt.subplot(gs[2, 4:9]), plt.subplot(gs[2, 9:])]]

    labels = [
        'a) 2022: excess ice 5--15 cm', 'b) 2022: excess ice 20--30 cm',
        'c) 2022: excess ice 45--55 cm', 'd) 2019: excess ice 5--15 cm',
        'e) 2019: excess ice 20--30 cm', 'f) 2019: excess ice 45--55 cm',
        'g) false-color image', 'h) 2022: transect T1', 'i) 2019: transect T1']

    for jyear, res in enumerate([res0, res1]):
        for jy, y in enumerate(ys):
            jy0, jy1 = _get_index(res['ygrid'], y[0]), _get_index(res['ygrid'], y[1])
            _e_mean = np.mean(res['e_mean'][..., jy0:jy1], axis=-1)
            # _e_mean[invalid] = np.nan
            ax = axs[jyear][jy]
            im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
            ax.set_facecolor('#aaaaaa')
            ax.set_xticks(xticks_im)
            ax.set_yticks(yticks_im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(color='#dddddd', linewidth=0.4)
    optical = res0['optical'][::-1, ...][0:3]
    ax = axs[-1][0]
    ax.imshow(contrast(np.moveaxis(optical, 0, -1)))
    ax.contour(
        res0['dem'][0, ...], colors=['#ffffff'], linewidths=0.4, alpha=0.4, levels=10)
    ax.set_xticks(np.array(xticks_im) * upscale)
    ax.set_yticks(np.array(yticks_im) * upscale)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='#aaaaaa', linewidth=0.4)
    pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])
    rc = pi._rowcol_endpoints
    label = f'T1'
    add_arrow_line(
        ax, rc, label=label, c='#ffffff', lw=0.7, alpha=0.9, dlabel=(95, -45))
    _xy_site = geospatial.upscaled(upscale).rowcol(site)[:, 0]
    ax.plot(
        _xy_site[1], _xy_site[0], c='#ffffff', linestyle='none',
        marker='o', ms=5, mfc='none')
    ax.text(_xy_site[1] + 30, _xy_site[0] - 40, 'HV', c='#ffffff')
    add_scalebar(ax, geospatial.upscaled(upscale), length=1000, label='1 km')

    xticks = [0, 250, 500, 750, 1000]
    yticks = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    ymax = 0.55

    plabels = [
        (0.13, 'inactive fp'), (0.45, 'abandoned fp'), (0.75, 'rocky'), (0.93, 'slope')]
    x_ylabel = -0.12
    plot_profile(
        axs[-1][1], res0['e_mean'], geospatial, profile, im_frac=res0['frac_thawed'],
        ymax=ymax, vlim=elim, ygrid=res0['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
        labels=None, x_ylabel=x_ylabel)
    axs[-1][1].text(0.05, 0.23, 'thaw depth', c='#ffffff', transform=axs[-1][1].transAxes)
    plot_profile(
        axs[-1][2], res1['e_mean'], geospatial, profile, im_frac=res1['frac_thawed'],
        ymax=ymax, vlim=elim, ygrid=res1['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
        labels=plabels, x_ylabel=x_ylabel)

    cax = axs[0][-1].inset_axes([1.17, -0.5, 0.15, 0.80])
    cax.text(1.0, 1.16, '$e$ [-]', ha='center', va='baseline', transform=cax.transAxes)
    plt.colorbar(im_e, cax, shrink=0.5, orientation='vertical', ticks=[0.0, 0.25, 0.50])

    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.010, 1.035, lab, ha='left', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout)
    #save

if __name__ == '__main__':
    from scripts.pathnames import paths
    fnplot = os.path.join(paths['figures'], 'happyvalley.pdf')
    happyvalley_map_profiles(fnout=fnplot)
