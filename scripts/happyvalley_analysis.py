'''
Created on Oct 5, 2022

@author: simon
'''

import numpy as np
from pathlib import Path

from analysis import load_object, save_object, InversionResults, read_K
from scripts.kivalina_analysis import resample_dem

site = np.array((-148.8437, 69.1548))[:, np.newaxis]

def path_results(year):
    pathres = Path(f'/home/simon/Work/gie/processed/Dalton_131_363/happyvalley/{year}/hadamard')
    return pathres

def invalid_mask(K, thresh, geospatial_K, geospatial, ind1=0, ind2=-1, wavelength=0.055):
    from scipy.ndimage import binary_dilation, binary_opening, binary_closing
    from analysis import add_atmospheric_K
    K = add_atmospheric_K(K, 0.0, wavelength=wavelength)
    K_last = K[ind1, ind1, ...] + K[ind2, ind2, ...] - 2 * K[ind1, ind2, ...]
    K_last_crop, _ = geospatial.warp(K_last, geospatial_K)
    s1 = np.array(
        [[ 0, 1, 0], [ 1, 1, 1], [0, 1, 0]])
    invalid = binary_closing(binary_opening(
        binary_dilation(K_last_crop > thresh ** 2, s1), s1, border_value=1), s1)
    # invalid = binary_dilation(binary_opening(binary_closing(K_last_crop > thresh ** 2, s1), s1), s1)
    return invalid

def read_results(pathres, fnimraw=None, fndemraw=None, upscale=8, overwrite=True):
    fngeospatial = pathres / 'geospatial.p'
    fnygrid = pathres / 'ygrid.p'
    if not fngeospatial.exists() or not fnygrid.exists() or overwrite:
        ir = InversionResults.from_file(pathres / 'ir.p')
        geospatial = ir.geospatial
        ygrid = ir.ygrid
        save_object(geospatial, fngeospatial)
        save_object(ygrid, fnygrid)
    else:
        geospatial = load_object(pathres / 'geospatial.p')
        ygrid = load_object(fnygrid)
    res = {'ygrid': ygrid, 'geospatial': geospatial}
    res['e_mean'] = np.load(pathres / 'e_mean.npy')
    res['e_quantile'] = np.load(pathres / 'e_quantile.npy')
    res['frac_thawed'] = np.load(pathres / 'frac_thawed_None.npy')

    if fnimraw is not None:
        fnimres = pathres / 'optical.tif'
        res['optical'] = resample_dem(
            geospatial, fnimraw, fnimres, upscale=upscale, overwrite=overwrite)
    if fndemraw is not None:
        fndemres = pathres / 'dem.tif'
        res['dem'] = resample_dem(
            geospatial, fndemraw, fndemres, upscale=upscale, overwrite=overwrite)
    return res

def happyvalley_map_profiles(fnout=None, overwrite=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scripts.plotting import (
        prepare_figure, cmap_e, colslist, _get_index, contrast, initialize_matplotlib,
        add_scalebar, plot_profile, add_arrow_line, ProfileInterpolator)
    years = (2022, 2019)
    fnimraw = Path('/home/simon/Work/gie/ancillary/Planet/20220620/20220620_211417_74_249d/'
                   'analytic_sr_udm2/20220620_211417_74_249d_3B_AnalyticMS_SR.tif')
    fndemraw = Path('/home/simon/Work/gie/ancillary/ArcticDEM/46_18_10m_v3.0_reg_dem.tif')
    path0 = Path('/home/simon/Work/gie/processed/Dalton_131_363/')
    wavelength, thresh = 0.055, 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    xticks_im = (25, 65, 105, 145)
    yticks_im = (25, 65, 105)
    ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]

    # profile = ((-148.8013, 69.1609), (-148.7717, 69.1609))
    profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))
    xy_ref = np.array([-148.8063, 69.1616])[:, np.newaxis]

    res0 = read_results(
        path_results(years[0]), fnimraw=fnimraw, fndemraw=fndemraw, upscale=upscale,
        overwrite=overwrite)
    res1 = read_results(path_results(years[1]), overwrite=overwrite)
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
        'c) 2022: excess ice 40--50 cm', 'd) 2019: excess ice 5--15 cm',
        'e) 2019: excess ice 20--30 cm', 'f) 2019: excess ice 40--50 cm',
        'g) false-color image', 'h) 2022: transect T1', 'i) 2019: transect T1']

    for jyear, res in enumerate([res0, res1]):
        fnK = path0 / str(years[jyear]) / 'K_vec.geo.tif'
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
    ax.imshow(contrast(np.moveaxis(optical, 0, -1), percentiles=(2.0, 92.0)))
    ax.contour(
        res0['dem'][0, ...], colors=['#ffffff'], linewidths=0.4, alpha=0.4, levels=10)
    rc_ref = np.array(geospatial.upscaled(upscale).rowcol(xy_ref))[:, 0]
    ax.plot(
        rc_ref[1], rc_ref[0], linestyle='none', ms=2.5, marker='x',
        mec=colslist[0], mfc=colslist[0], zorder=9)
    ax.set_xticks(np.array(xticks_im) * upscale)
    ax.set_yticks(np.array(yticks_im) * upscale)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='#aaaaaa', linewidth=0.4)
    pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])
    rc = pi._rowcol_endpoints
    label = f'T1'
    add_arrow_line(
        ax, rc, label=label, c='#ffffff', lw=0.7, alpha=0.9, dlabel=(380, -180), hwidth=140,
        hlength=180)
    _xy_site = geospatial.upscaled(upscale).rowcol(site)[:, 0]
    ax.plot(
        _xy_site[1], _xy_site[0], c='#ffffff', linestyle='none',
        marker='o', ms=5, mfc='none')
    ax.text(_xy_site[1] + 120, _xy_site[0] - 160, 'HV', c='#ffffff')
    add_scalebar(ax, geospatial.upscaled(upscale), length=1000, label='1 km')
    ax.text(0.01, -0.06, 'Planet Labs', ha='left', va='top', transform=ax.transAxes)

    xticks = [0, 250, 500, 750, 1000]
    yticks = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    ymax = 0.50

    plabels = [
        (0.13, 'inactive fp'), (0.45, 'abandoned fp'), (0.75, 'rocky'), (0.93, 'slope')]
    x_ylabel = -0.12
    plot_profile(
        axs[-1][1], res0['e_mean'], geospatial, profile, im_frac=res0['frac_thawed'],
        ymax=ymax, vlim=elim, ygrid=res0['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
        labels=None, x_ylabel=x_ylabel)
    axs[-1][1].text(
        0.05, 0.19, '$y_\\mathrm{f}$', c='#ffffff', transform=axs[-1][1].transAxes, alpha=0.6)
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
        plt.savefig(fnout, dpi=450)

def happyvalley_map_subsidence(fnout=None, overwrite=True):
    import matplotlib.pyplot as plt
    from scripts.northslope_comparison import read_InSAR
    from scripts.plotting import (
        prepare_figure, cmap_e, cmap_s, _get_index, add_scalebar,)
    site = 'happyvalley'
    years = (2022, 2019)
    path0 = Path('/home/simon/Work/gie/processed/Dalton_131_363/')
    wavelength, thresh = 0.055, 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    slim = (-0.06, 0.06)
    xticks_im = (25, 65, 105, 145)
    yticks_im = (25, 65, 105)
    ys = (0.40, 0.50)

    xy_ref = np.array([-148.8063, 69.1616])[:, np.newaxis]

    res0 = read_results(
        path_results(years[0]), upscale=upscale, overwrite=overwrite)
    res1 = read_results(path_results(years[1]), overwrite=overwrite)
    resl = [res0, res1]
    sresl = [read_InSAR(site, year) for year in years]
    geospatial = res0['geospatial']
    assert res1['geospatial'] == geospatial

    fig, axs = prepare_figure(
        ncols=3, nrows=2, figsize=(1.70, 0.90), sharex=True, sharey=True, left=0.04, right=0.98,
        bottom=0.14, top=0.95, hspace=0.26, wspace=0.10, remove_spines=False)
    inds = ([(4, -1), (0, -1)], [(5, -1), (0, -1)])
    labels = [
        'a) excess ice 40--50 cm', 'b) $s$ Jul 28 -- Sep 14, 2022',
        'c) $s$ Jun 10 -- Sep 14, 2022', 'd) excess ice 40--50 cm',
        'e) $s$ Aug 01 -- Sep 06, 2019', 'f) $s$ Jun 02 -- Sep 06, 2019', ]

    for jyear, year in enumerate(years):
        fnK = path0 / str(years[jyear]) / 'K_vec.geo.tif'
        K, geospatial_K = read_K(fnK)
        invalid = invalid_mask(
            K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)

        jy0, jy1 = _get_index(resl[jyear]['ygrid'], ys[0]), _get_index(resl[jyear]['ygrid'], ys[1])
        _e_mean = np.mean(resl[jyear]['e_mean'][..., jy0:jy1], axis=-1)
        _e_mean[invalid] = np.nan
        ax = axs[jyear, 0]
        im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
        s_obs_v = sresl[jyear]['s_obs'] / np.cos(sresl[jyear]['geom']['ia'])
        s_obs_v = np.concatenate((np.zeros((1,) + s_obs_v.shape[1:]), s_obs_v), axis=0)
        for jind, ind in enumerate(inds[jyear]):
            s_obs_diff = s_obs_v[ind[1], ...] - s_obs_v[ind[0], ...]
            print(sresl[jyear]['dates'][ind[0]], sresl[jyear]['dates'][ind[1]])
            s_obs_diff_crop, _ = geospatial.warp(s_obs_diff, sresl[jyear]['geospatial'])
            s_obs_diff_crop[invalid] = np.nan
            ax = axs[jyear, jind + 1]
            im_s = ax.imshow(s_obs_diff_crop, cmap=cmap_s, vmin=slim[0], vmax=slim[1])

    for ax in axs.flatten():
        ax.set_facecolor('#aaaaaa')
        ax.set_xticks(xticks_im)
        ax.set_yticks(yticks_im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='#dddddd', linewidth=0.4, alpha=0.5)

    rc_ref = np.array(geospatial.rowcol(xy_ref))[:, 0]
    ax.plot(
        rc_ref[1], rc_ref[0], linestyle='none', ms=2.5, marker='x',
        mec='#333333', zorder=9)
    add_scalebar(ax, geospatial, length=1000, label='1 km')

    cparms = {'e': ('$e$ [-]', (0.0, 0.25, 0.50), None),
              's': ('$s$ [cm]', (-0.05, 0.0, 0.05), (-5, 0, 5))}
    ims = (im_e, im_s)
    for jvar, _var in enumerate(['e', 's']):
        _cp = cparms[_var]
        cax = axs[-1, jvar].inset_axes([0.1, -0.22, 0.60, 0.12])
        cax.text(1.08, 0.20, _cp[0], ha='left', va='baseline', transform=cax.transAxes)
        cbar = plt.colorbar(ims[jvar], cax, shrink=0.5, orientation='horizontal', ticks=_cp[1])
        if _cp[2] is not None:
            cbar.set_ticks(_cp[1], labels=_cp[2])
        cbar.solids.set_rasterized(True)

    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.010, 1.035, lab, ha='left', va='baseline', transform=ax.transAxes)
    for jax, ax in enumerate(axs[:, 0]):
        ax.text(-0.10, 0.50, years[jax], rotation=90, ha='right', va='center', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout, dpi=450)

def happyvalley_map_profiles_2023(fnout=None, overwrite=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scripts.plotting import (
        prepare_figure, cmap_e, colslist, _get_index, contrast, initialize_matplotlib,
        add_scalebar, plot_profile, add_arrow_line, ProfileInterpolator)
    years = (2023, 2022)
    fnimraw = Path('/home/simon/Work/gie/ancillary/Planet/20220620/20220620_211417_74_249d/'
                   'analytic_sr_udm2/20220620_211417_74_249d_3B_AnalyticMS_SR.tif')
    fndemraw = Path('/home/simon/Work/gie/ancillary/ArcticDEM/46_18_10m_v3.0_reg_dem.tif')
    path0 = Path('/home/simon/Work/gie/processed/Dalton_131_363/')
    wavelength, thresh = 0.055, 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    xticks_im = (25, 65, 105, 145)
    yticks_im = (25, 65, 105)
    ys = [(0.00, 0.20), (0.20, 0.40), (0.40, 0.60)]

    # profile = ((-148.8013, 69.1609), (-148.7717, 69.1609))
    profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))
    xy_ref = np.array([-148.8063, 69.1616])[:, np.newaxis]

    res0 = read_results(
        path_results(years[0]), fnimraw=fnimraw, fndemraw=fndemraw, upscale=upscale,
        overwrite=overwrite)
    res1 = read_results(path_results(years[1]), overwrite=overwrite)
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
        'a) 2023: excess ice 0--20 cm', 'b) 2023: excess ice 20--40 cm',
        'c) 2023: excess ice 40--60 cm', 'd) 2022: excess ice 0--20 cm',
        'e) 2022: excess ice 20--40 cm', 'f) 2022: excess ice 40--60 cm',
        'g) false-color image', 'h) 2023: transect T1', 'i) 2022: transect T1']

    for jyear, res in enumerate([res0, res1]):
        fnK = path0 / str(years[jyear]) / 'K_vec.geo.tif'
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
    ax.imshow(contrast(np.moveaxis(optical, 0, -1), percentiles=(2.0, 92.0)))
    ax.contour(
        res0['dem'][0, ...], colors=['#ffffff'], linewidths=0.4, alpha=0.4, levels=10)
    rc_ref = np.array(geospatial.upscaled(upscale).rowcol(xy_ref))[:, 0]
    ax.plot(
        rc_ref[1], rc_ref[0], linestyle='none', ms=2.5, marker='x',
        mec=colslist[0], mfc=colslist[0], zorder=9)
    ax.set_xticks(np.array(xticks_im) * upscale)
    ax.set_yticks(np.array(yticks_im) * upscale)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='#aaaaaa', linewidth=0.4)
    pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])
    rc = pi._rowcol_endpoints
    label = f'T1'
    add_arrow_line(
        ax, rc, label=label, c='#ffffff', lw=0.7, alpha=0.9, dlabel=(380, -180), hwidth=140,
        hlength=180)
    _xy_site = geospatial.upscaled(upscale).rowcol(site)[:, 0]
    ax.plot(
        _xy_site[1], _xy_site[0], c='#ffffff', linestyle='none',
        marker='o', ms=5, mfc='none')
    ax.text(_xy_site[1] + 120, _xy_site[0] - 160, 'HV', c='#ffffff')
    add_scalebar(ax, geospatial.upscaled(upscale), length=1000, label='1 km')
    ax.text(0.01, -0.06, 'Planet Labs', ha='left', va='top', transform=ax.transAxes)

    xticks = [0, 250, 500, 750, 1000]
    yticks = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    ymax = 0.60

    plabels = [
        (0.13, 'inactive fp'), (0.45, 'abandoned fp'), (0.75, 'rocky'), (0.93, 'slope')]
    x_ylabel = -0.12
    plot_profile(
        axs[-1][1], res0['e_mean'], geospatial, profile, im_frac=res0['frac_thawed'],
        ymax=ymax, vlim=elim, ygrid=res0['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
        labels=None, x_ylabel=x_ylabel)
    axs[-1][1].text(
        0.05, 0.19, '$y_\\mathrm{f}$', c='#ffffff', transform=axs[-1][1].transAxes, alpha=0.6)
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
    # save

if __name__ == '__main__':
    from scripts.pathnames import paths
    fnplot = paths['figures'] / 'happyvalley.pdf'
    # happyvalley_map_profiles(fnout=fnplot, overwrite=False)
    fnplot = paths['figures'] / 'happyvalley23.pdf'
    # happyvalley_map_profiles_2023(fnout=fnplot, overwrite=False)
    fnplot = paths['figures'] / 'happyvalley_subs.pdf'
    happyvalley_map_subsidence(fnplot, overwrite=False)

