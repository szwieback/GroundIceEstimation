'''
Created on Oct 5, 2022

@author: simon
'''
import numpy as np
import os
from analysis import (
    Geospatial, load_object, save_object, InversionResultsIS, read_K, save_geotiff)
from scripts.kivalina_analysis import resample_dem

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.plotting import (
    prepare_figure, cmap_e, colslist, _get_index, contrast, initialize_matplotlib,
    add_scalebar, plot_profile, add_arrow_line, ProfileInterpolator)

site = np.array([-156.7723, 71.2825])[:, np.newaxis]

# def path_results(year):
#     pathres = f'/export/data/Experiments/gie/processed/Utqiagvik/{year}/hadamard'
#     return pathres
def path_results(sensor='s1', year='2023', rmethod='hadamard'):
    pathres = f'/export/data/Experiments/gie/processed/utqiagvik/{sensor}/{year}/{rmethod}'
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
    fngeospatial = os.path.join(pathres, 'geospatial.p')
    fnygrid = os.path.join(pathres, 'ygrid.p')

    if not os.path.exists(fngeospatial) or not os.path.exists(fnygrid) or overwrite:
        ir = InversionResultsIS.from_file(os.path.join(pathres, 'ir.p'))
        geospatial = ir.geospatial
        ygrid = ir.ygrid
        save_object(geospatial, fngeospatial)
        save_object(ygrid, fnygrid)
    else:
        geospatial = load_object(os.path.join(pathres, 'geospatial.p'))
        ygrid = load_object(fnygrid)
    res = {'ygrid': ygrid, 'geospatial': geospatial}
    res['e_mean'] = np.load(os.path.join(pathres, 'e_mean.npy'))
    # res['e_quantile'] = np.load(os.path.join(pathres, 'e_quantile.npy'))
    res['frac_thawed'] = np.load(os.path.join(pathres, 'frac_thawed_None.npy'))

    if fnimraw is not None:
        fnimres = os.path.join(pathres, 'optical.tif')
        res['optical'] = resample_dem(
            geospatial, fnimraw, fnimres, upscale=upscale, overwrite=overwrite)
    if fndemraw is not None:
        fndemres = os.path.join(pathres, 'dem.tif')
        res['dem'] = resample_dem(
            geospatial, fndemraw, fndemres, upscale=upscale, overwrite=overwrite)
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
def utqiagvik_map(years, sensor='s1', rmethod='hadamard', wavelength=0.055, fnout=None, overwrite=True):
    thresh = 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    # xticks_im = (25, 65, 105, 145)
    # yticks_im = (25, 65, 105)
    # ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]
    ys = [(0.05, 0.25), (0.25, 0.40)]

    profile = ((-149.8916,70.4955), (-149.8867,70.4967))
    # profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))
    # profile = ((392260, 7823681), (392511,7823824))
    # xy_ref = np.array([-149.8504,70.4836])[:, np.newaxis]

    res0 = read_results(path_results(sensor=sensor, year=years[0], rmethod=rmethod), overwrite=overwrite)
    res1 = read_results(path_results(sensor=sensor, year=years[1], rmethod=rmethod), overwrite=overwrite)
    geospatial = res0['geospatial']
    print(geospatial)
    # assert res1['geospatial'] == geospatial

    # fig, axs = prepare_figure(ncols=2, nrows=3, sharex='none', sharey='none')
    initialize_matplotlib()
    fig = plt.figure()
    fig.set_size_inches((4.50, 3.85), forward=True)
    gs = gridspec.GridSpec(
        2, 8, left=0.007, right=0.85, top=0.95, bottom=0.1, wspace=0.2, hspace=0.4)
    axs = [
        [fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:8])],
        [fig.add_subplot(gs[1, 0:4]), fig.add_subplot(gs[1, 4:8])],
        # [fig.add_subplot(gs[2, 1:7]), fig.add_subplot(gs[2, 7:8])]
    ]
    # axs[-1][-1].set_axis_off()
    labels = [
        f'a) 2023: excess ice {str(int(ys[0][0]*100))}--{str(int(ys[0][1]*100))} cm',
        f'b) 2023: excess ice {str(int(ys[1][0]*100))}--{str(int(ys[1][1]*100))} cm',
        f'c) 2024: excess ice {str(int(ys[0][0]*100))}--{str(int(ys[0][1]*100))} cm',
        f'd) 2024: excess ice {str(int(ys[1][0]*100))}--{str(int(ys[1][1]*100))} cm']

    for jyear, res in enumerate([res0, res1]):
        # K, geospatial_K = read_K(fnK)
        # invalid = invalid_mask(
        #     K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)
        for jy, y in enumerate(ys):
            jy0, jy1 = _get_index(res['ygrid'], y[0]), _get_index(res['ygrid'], y[1])
            _e_mean = np.mean(res['e_mean'][..., jy0:jy1], axis=-1)
            # _e_mean[invalid] = np.nan
            fnout_emean = os.path.join(fnout_dir, f'emean_{years[jyear]}_{int(y[0] * 100)}-{int(y[1] * 100)}.tif')
            # fnout_emean = os.path.join(fnout_dir, f'emean_{years}_{int(y[0]*100)}-{int(y[1]*100)}.tif')
            save_geotiff(np.expand_dims(_e_mean, axis=0), geospatial, fnout=fnout_emean)
            ax = axs[jyear][jy]
            # crop_edges = True
            # if crop_edges:
            #     row_start, row_end, col_start, col_end = crop_nan_edges(_e_mean)
            #     print(_e_mean.shape)
            #     print(row_start, row_end, col_start, col_end)
            #     raise
            #     squeezed_emean = _e_mean[row_start:row_end + 1, col_start:col_end + 1]
            #     _e_mean = squeezed_emean
            im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
            ax.set_facecolor('#aaaaaa')
            # ax.set_xticks(xticks_im)
            # ax.set_yticks(yticks_im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(color='#dddddd', linewidth=0.4, alpha=0.5)
    add_scalebar(axs[0][0], geospatial.upscaled(upscale), length=1000, label='1 km')
    # optical = res0['optical'][::-1, ...][0:3]
    # ax = axs[-1][0]
    # ax.imshow(contrast(np.moveaxis(optical, 0, -1), percentiles=(2.0, 92.0)))
    # ax.contour(
    #     res0['dem'][0, ...], colors=['#ffffff'], linewidths=0.4, alpha=0.4, levels=10)
    # rc_ref = np.array(geospatial.upscaled(upscale).rowcol(xy_ref))[:, 0]
    # ax.plot(
    #     rc_ref[1], rc_ref[0], linestyle='none', ms=2.5, marker='x',
    #     mec=colslist[0], mfc=colslist[0], zorder=9)
    # ax.set_xticks(np.array(xticks_im) * upscale)
    # ax.set_yticks(np.array(yticks_im) * upscale)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.grid(color='#aaaaaa', linewidth=0.4)
    # pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])

    # rc = pi._rowcol_endpoints
    # label = f'T1'
    # add_arrow_line(
    #     ax, rc, label=label, c='#ffffff', lw=0.7, alpha=0.9, dlabel=(380, -180), hwidth=140,
    #     hlength=180)
    # _xy_site = geospatial.upscaled(upscale).rowcol(site)[:, 0]
    # ax.plot(
    #     _xy_site[1], _xy_site[0], c='#ffffff', linestyle='none',
    #     marker='o', ms=5, mfc='none')
    # ax.text(_xy_site[1] + 120, _xy_site[0] - 160, 'HV', c='#ffffff')

    # ax.text(0.01, -0.06, 'Planet Labs', ha='left', va='top', transform=ax.transAxes)

    xticks = [0, 100, 200]
    yticks = (0.0, 0.1, 0.2, 0.3, 0.4)
    ymax = 0.45

    # plabels = [
    #     (0.13, 'inactive fp'), (0.45, 'abandoned fp'), (0.75, 'rocky'), (0.93, 'slope')]
    x_ylabel = -0.12

    # plot_profile(
    #     axs[2][0], res0['e_mean'], geospatial, profile, im_frac=res0['frac_thawed'],
    #     ymax=ymax, vlim=elim, ygrid=res0['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
    #     labels=None, x_ylabel=x_ylabel)
    # axs[2][0].text(
    #     0.05, 0.19, '$y_\\mathrm{f}$', c='#ffffff', transform=axs[-1][1].transAxes, alpha=0.6)
    # plot_profile(
    #     axs[-1][2], res1['e_mean'], geospatial, profile, im_frac=res1['frac_thawed'],
    #     ymax=ymax, vlim=elim, ygrid=res1['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
    #     labels=plabels, x_ylabel=x_ylabel)
    #
    cax = axs[0][-1].inset_axes([1.2, -0.5, 0.15, 0.80])
    cax.text(1.0, 1.16, '$e$ [-]', ha='center', va='baseline', transform=cax.transAxes)
    plt.colorbar(im_e, cax, shrink=0.5, orientation='vertical', ticks=[0.0, 0.25, 0.50])

    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.010, 1.035, lab, ha='left', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout, dpi=300)
        plt.show()

def Utqiagvik_map_profiles(fnout=None, overwrite=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scripts.plotting import (
        prepare_figure, cmap_e, colslist, _get_index, contrast, initialize_matplotlib,
        add_scalebar, plot_profile, add_arrow_line, ProfileInterpolator)
    years = (2023, 2024)
    # fnimraw = '/home/simon/Work/gie/ancillary/Planet/20220620/20220620_211417_74_249d/analytic_sr_udm2/20220620_211417_74_249d_3B_AnalyticMS_SR.tif'
    # fndemraw = '/home/simon/Work/gie/ancillary/ArcticDEM/46_18_10m_v3.0_reg_dem.tif'
    # path0 = '/home/simon/Work/gie/processed/Dalton_131_363/'
    wavelength, thresh = 0.055, 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    xticks_im = (25, 65, 105, 145)
    yticks_im = (25, 65, 105)
    # ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]
    ys = [(0.05, 0.25), (0.25, 0.40)]

    # profile = ((-148.8013, 69.1609), (-148.7717, 69.1609))
    # profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))
    # profile = ((-149.90322, 70.48570), (-149.89026, 70.48857))
    # xy_ref = np.array([-149.8504,70.4836])[:, np.newaxis]

    # res0 = read_results(
    #     path_results(years[0]), fnimraw=fnimraw, fndemraw=fndemraw, upscale=upscale,
    #     overwrite=overwrite)
    # res0 = read_results(path_results(years[0]), upscale=upscale, overwrite=overwrite)
    res0 = read_results(path_results(years[0]), overwrite=overwrite)
    res1 = read_results(path_results(years[1]), overwrite=overwrite)
    # print(res0)
    print(res0['e_mean'][:, :, 350])
    print(res1['e_mean'][:, :, 150])
    geospatial = res0['geospatial']
    assert res1['geospatial'] == geospatial

    # fig, axs = prepare_figure(ncols=3, nrows=3, sharex='none', sharey='none')
    fig = plt.figure()
    initialize_matplotlib()
    fig.set_size_inches((4.50, 3.85), forward=True)
    gs = gridspec.GridSpec(
        2, 8, left=0.007, right=0.85, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
    axs = [[plt.subplot(gs[0, 0:4]), plt.subplot(gs[0, 4:8])],
           [plt.subplot(gs[1, 0:4]), plt.subplot(gs[1, 4:8])]]

    labels = [
        f'a) 2023: excess ice {str(int(ys[0][0]*100))}--{str(int(ys[0][1]*100))} cm',
        f'b) 2023: excess ice {str(int(ys[1][0]*100))}--{str(int(ys[1][1]*100))} cm',
        f'c) 2024: excess ice {str(int(ys[0][0]*100))}--{str(int(ys[0][1]*100))} cm',
        f'd) 2024: excess ice {str(int(ys[1][0]*100))}--{str(int(ys[1][1]*100))} cm']

    for jyear, res in enumerate([res0, res1]):
        # fnK = os.path.join(path0, str(years[jyear]), 'K_vec.geo.tif')
        fnK = f'/export/data/Experiments/stacks/OliktokPoint_P102D/{years[jyear]}/proc/hadamard/geocoded/K_vec.geo.tif'
        K, geospatial_K = read_K(fnK)
        invalid = invalid_mask(
            K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)
        for jy, y in enumerate(ys):
            jy0, jy1 = _get_index(res['ygrid'], y[0]), _get_index(res['ygrid'], y[1])
            _e_mean = np.mean(res['e_mean'][..., jy0:jy1], axis=-1)
            _e_mean[invalid] = np.nan
            fnout_emean = os.path.join(fnout_dir, f'emean_{years[jyear]}_{int(y[0] * 100)}-{int(y[1] * 100)}.tif')
            save_geotiff(np.expand_dims(_e_mean, axis=0), geospatial, fnout=fnout_emean)
            ax = axs[jyear][jy]
            im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
            ax.set_facecolor('#aaaaaa')
            ax.set_xticks(xticks_im)
            ax.set_yticks(yticks_im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(color='#dddddd', linewidth=0.4, alpha=0.5)
    # optical = res0['optical'][::-1, ...][0:3]
    # ax = axs[-1][0]
    # ax.imshow(contrast(np.moveaxis(optical, 0, -1), percentiles=(2.0, 92.0)))
    # ax.contour(
    #     res0['dem'][0, ...], colors=['#ffffff'], linewidths=0.4, alpha=0.4, levels=10)
    # rc_ref = np.array(geospatial.upscaled(upscale).rowcol(xy_ref))[:, 0]
    # ax.plot(
    #     rc_ref[1], rc_ref[0], linestyle='none', ms=2.5, marker='x',
    #     mec=colslist[0], mfc=colslist[0], zorder=9)
    # ax.set_xticks(np.array(xticks_im) * upscale)
    # ax.set_yticks(np.array(yticks_im) * upscale)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.grid(color='#aaaaaa', linewidth=0.4)
    # pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])
    # rc = pi._rowcol_endpoints
    label = f'T1'
    # add_arrow_line(
    #     ax, rc, label=label, c='#ffffff', lw=0.7, alpha=0.9, dlabel=(380, -180), hwidth=140,
    #     hlength=180)
    # _xy_site = geospatial.upscaled(upscale).rowcol(site)[:, 0]
    # ax.plot(
    #     _xy_site[1], _xy_site[0], c='#ffffff', linestyle='none',
    #     marker='o', ms=5, mfc='none')
    # ax.text(_xy_site[1] + 120, _xy_site[0] - 160, 'HV', c='#ffffff')
    # add_scalebar(ax, geospatial.upscaled(upscale), length=1000, label='1 km')
    # ax.text(0.01, -0.06, 'Planet Labs', ha='left', va='top', transform=ax.transAxes)

    # xticks = [0, 250, 500, 750, 1000]
    # yticks = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    # ymax = 0.50
    #
    # plabels = [
    #     (0.13, 'inactive fp'), (0.45, 'abandoned fp'), (0.75, 'rocky'), (0.93, 'slope')]
    # x_ylabel = -0.12
    # plot_profile(
    #     axs[-1][1], res0['e_mean'], geospatial, profile, im_frac=res0['frac_thawed'],
    #     ymax=ymax, vlim=elim, ygrid=res0['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
    #     labels=None, x_ylabel=x_ylabel)
    # axs[-1][1].text(
    #     0.05, 0.19, '$y_\\mathrm{f}$', c='#ffffff', transform=axs[-1][1].transAxes, alpha=0.6)
    # plot_profile(
    #     axs[-1][2], res1['e_mean'], geospatial, profile, im_frac=res1['frac_thawed'],
    #     ymax=ymax, vlim=elim, ygrid=res1['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
    #     labels=plabels, x_ylabel=x_ylabel)

    cax = axs[0][-1].inset_axes([1.05, -0.5, 0.15, 0.80])
    cax.text(1.0, 1.16, '$e$ [-]', ha='center', va='baseline', transform=cax.transAxes)
    plt.colorbar(im_e, cax, shrink=0.5, orientation='vertical', ticks=[0.0, 0.25, 0.50])

    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.010, 1.035, lab, ha='left', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout, dpi=300)
        plt.show()
    # save

if __name__ == '__main__':
    years = (2023, 2024)
    site_name = 'utqiagvik'
    # site_name = 'oliktok'
    sensor = 's1'
    # stack_method = 'hadamard'
    stack_method = 'mintpy'
    fnout_dir = f'/export/data/Experiments/gie/processed/{site_name}/postproc/'
    fig_dir = f'/home/jchen20/Dropbox/figures_paper/stacks/{site_name}/Oliktok_{sensor}_{stack_method}.png'

    # check_results(years[0], sensor='s1', rmethod='hadamard', wavelength=0.055, overwrite=True)
    utqiagvik_map(years=years, sensor=sensor, rmethod=stack_method, wavelength=0.055, fnout=fig_dir, overwrite=False)



    # from scripts.pathnames import paths
    # site_name = 'Utqiagvik'
    # site_name = 'Utqiagvik'
    # fnout_dir = '/export/data/Experiments/gie/processed/Utqiagvik/postproc/'
    # fig_dir = f'/home/jchen20/Dropbox/figures_paper/stacks/{site_name}/Utqiagvik.png'
    # Utqiagvik_map_profiles(fnout=fig_dir, overwrite=False)
