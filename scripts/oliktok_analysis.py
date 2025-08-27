#!/usr/bin/env python
'''
Created on Oct 5, 2022

@author: simon
'''
import numpy as np
import os
from pathlib import Path
import rasterio
from numpy.ma.core import squeeze

from analysis import (
    Geospatial, load_object, save_object, InversionResultsIS, read_K, save_geotiff)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.plotting import (
    prepare_figure, cmap_e, colslist, _get_index, contrast, initialize_matplotlib,
    add_scalebar, plot_profile, add_arrow_line, ProfileInterpolator)

from scripts.kivalina_analysis import resample_dem

site = np.array((-149.886728,70.496709))[:, np.newaxis]

def print_matrix(matrix):
    for row in matrix:
        print(" ".join("{:.5f}".format(x) for x in row))

def process_rgb_image(red_band, green_band, blue_band, normalize=True, alpha=1.0, beta=0.0, gamma=1.0):
    # https://www.satmapper.hu/en/rgb-images/
    # Additional details: https://geospatialyst.readthedocs.io/en/latest/Content/Lesson/geo-python-course/06.Raster-data-analysis.html
    def normalize_band(band):
        band_min, band_max = band.min(), band.max()
        return (band - band_min) / (band_max - band_min)

    if normalize:
        red_band = normalize_band(red_band)
        green_band = normalize_band(green_band)
        blue_band = normalize_band(blue_band)

    # alpha: Brightness scaling factor; beta: Brightness offset
    if alpha != 1.0 or beta != 0.0:
        red_band = np.clip(alpha * red_band + beta, 0, 1)
        green_band = np.clip(alpha * green_band + beta, 0, 1)
        blue_band = np.clip(alpha * blue_band + beta, 0, 1)

    # Gamma correction. The math behind it is that we take each pixels
    # intesnity values and raise it to the power of (1/gamma) where the gamma value is specified by us.
    if gamma != 1.0:  # Gamma correction factor
        red_band = np.power(red_band, 1 / gamma)
        green_band = np.power(green_band, 1 / gamma)
        blue_band = np.power(blue_band, 1 / gamma)

    # Normalize again if brightness or gamma correction was applied
    if normalize and (alpha != 1.0 or beta != 0.0 or gamma != 1.0):
        red_band = normalize_band(red_band)
        green_band = normalize_band(green_band)
        blue_band = normalize_band(blue_band)
    rgb_composite = np.stack((red_band, green_band, blue_band), axis=0)
    # rgb_composite = (rgb_composite * 255).astype(np.uint8)
    return rgb_composite

def crop_nan_edges(arr):
    valid_rows = ~np.isnan(arr).all(axis=1)
    print(valid_rows.shape)
    raise
    row_start = np.argmax(valid_rows)
    row_end = len(valid_rows) - np.argmax(valid_rows[::-1]) - 1
    valid_cols = ~np.isnan(arr).all(axis=0)
    col_start = np.argmax(valid_cols)
    col_end = len(valid_cols) - np.argmax(valid_cols[::-1]) - 1
    return row_start, row_end, col_start, col_end

def path_results(sensor='s1', year='2023', rmethod='hadamard', resolution='40m'):
    pathres = f'/export/data/Experiments/gie/processed/{site_name}/{sensor}/{year}/{rmethod}/{resolution}'
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
def oliktok_map(years, sensor='s1', rmethod='hadamard', resolution='40m', fnout=None, overwrite=True):
    thresh = 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    # xticks_im = (25, 65, 105, 145)
    # yticks_im = (25, 65, 105)
    # ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]
    ys = [(0.05, 0.25), (0.25, 0.50)]

    profile = ((-149.8916,70.4955), (-149.8867,70.4967))
    # profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))
    # profile = ((392260, 7823681), (392511,7823824))
    # xy_ref = np.array([-149.8504,70.4836])[:, np.newaxis]
    # resolution = '40m'
    res0 = read_results(path_results(sensor=sensor, year=years[0], rmethod=rmethod, resolution=resolution), overwrite=overwrite)
    res1 = read_results(path_results(sensor=sensor, year=years[1], rmethod=rmethod, resolution=resolution), overwrite=overwrite)
    geospatial = res0['geospatial']
    print(geospatial)
    # assert res1['geospatial'] == geospatial

    # fig, axs = prepare_figure(ncols=2, nrows=3, sharex='none', sharey='none')
    initialize_matplotlib()
    fig = plt.figure()
    fig.set_size_inches((3.8, 3), forward=True)
    gs = gridspec.GridSpec(
        2, 8, left=0.007, right=0.85, top=0.95, bottom=0.1, wspace=0.1, hspace=0.4)
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
            print(jy)
            jy0, jy1 = _get_index(res['ygrid'], y[0]), _get_index(res['ygrid'], y[1])
            _e_mean = np.mean(res['e_mean'][..., jy0:jy1], axis=-1)
            # _e_mean[invalid] = np.nan
            # fnout_emean = os.path.join(fnout_dir, f'emean_{years[jyear]}_{int(y[0]*100)}-{int(y[1]*100)}.tif')
            fnout_emean = fnout_dir / f'emean_{years[jyear]}_{int(y[0] * 100)}-{int(y[1] * 100)}.tif'
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
    pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])

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
    cax = axs[0][-1].inset_axes([1.05, 0.1, 0.15, 0.80])
    cax.text(1.0, 1.16, '$e$ [-]', ha='center', va='baseline', transform=cax.transAxes)
    plt.colorbar(im_e, cax, shrink=0.5, orientation='vertical', ticks=[0.0, 0.25, 0.50])

    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.010, 1.035, lab, ha='left', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout, dpi=300)
        plt.show()

# def pts_defo(ax, defo, date_list, ylim,
#                            defo_perc25=None, defo_perc75=None, color='#1E90FF', fill_color='#ADD8E6',
#                            legend_label='', fig_title=None):
#     ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
#
#     if defo_perc25 is not None and defo_perc75 is not None:
#         ax.fill_between(date_list, defo_perc10.flatten(), defo_perc25.flatten(), color=fill_color, alpha=0.5)
#         ax.fill_between(date_list, defo_perc75.flatten(), defo_perc90.flatten(), color=fill_color, alpha=0.5)
#         ax.fill_between(date_list, defo_perc25.flatten(), defo_perc75.flatten(), color=color, alpha=0.5)
#
#     ax.plot(date_list, defo_median, marker='o', markersize=8, markerfacecolor=color, markeredgecolor='black',
#             linestyle='-', color='black', label=legend_label, linewidth=2)
#     # start_date = mdates.datestr2num('12/31/2016')
#     # end_date = mdates.datestr2num('12/31/2022')
#     # ax.set_xlim([start_date, end_date])
#     # ax = set_x_date(ax, nmonths=12)
#
#     y_ticks = np.arange(ylim[0] + 1, 0.1, step=4)
#     ax.set_ylim(ylim)
#     ax.set_yticks(y_ticks)
#     # ax.set_xlabel('Years')
#     ax.set_ylabel('Interannual deformation ($\hat{d}_{e}$)')
#     if fig_title is not None:
#         ax.set_title(fig_title, loc='left')
#     return ax
def get_rowcol(transform, points):
    if points.ndim == 1:
        r, c = rasterio.transform.rowcol(transform, [points[0]], [points[1]])
        return np.array([r[0], c[0]])
    elif points.ndim == 2:
        r, c = rasterio.transform.rowcol(transform, points[:, 0], points[:, 1])
        return np.stack((r, c), axis=1)
    else:
        raise ValueError("Invalid input shape. 'points' must be a single point or an array of points.")

def get_point_h5(fn_h5, x, y, ds_name='e_mean', coord_crs=None, raster_crs=None):
    import h5py
    import numpy as np
    from rasterio.crs import CRS
    from rasterio.transform import Affine, rowcol
    from pyproj import Transformer
    with h5py.File(fn_h5, 'r') as f:
        a = f['/'].attrs
        x_first = float(a['X_FIRST']); y_first = float(a['Y_FIRST'])
        x_step  = float(a['X_STEP']);  y_step  = float(a['Y_STEP'])  # often negative
        transform = Affine(x_step, 0.0, x_first, 0.0, y_step, y_first)

        crs = None
        if 'EPSG' in a:
            try:
                code = int(str(a['EPSG']).split(':')[-1])
                crs = CRS.from_epsg(code)
            except Exception:
                crs = None
        if crs is None and 'CRS_WKT' in a:
            wkt = a['CRS_WKT']
            if isinstance(wkt, (bytes, bytearray)):
                wkt = wkt.decode('utf-8', 'ignore')
            if isinstance(wkt, str):
                wkt = wkt.replace('&quot;', '"')
                try:
                    crs = CRS.from_wkt(wkt)
                except Exception:
                    crs = None
        if crs is None and raster_crs is not None:
            crs = CRS.from_string(raster_crs)
        if crs is None:
            raise ValueError("Raster CRS not found in HDF5 and no raster_crs provided.")

        if coord_crs is not None:
            in_crs = CRS.from_string(coord_crs)
            if in_crs != crs:
                T = Transformer.from_crs(in_crs, crs, always_xy=True)
                x, y = T.transform(x, y)

        r, c = rowcol(transform, x, y, op=np.floor)
        r, c = int(r), int(c)
        print(r, c)

        dset = f[ds_name]
        if dset.ndim == 3:   # (rows, cols, bands)
            rows, cols, bands = dset.shape
        elif dset.ndim == 2: # (rows, cols)
            rows, cols = dset.shape
            bands = 1
        else:
            raise ValueError("Unsupported dataset rank; expected 2D or 3D.")

        if not (0 <= r < rows and 0 <= c < cols):
            raise IndexError("Point is outside raster extent.")

        if bands == 1:
            values = float(dset[r, c])
            z_axis = None
        else:
            values = np.array(dset[r, c, :], dtype=np.float32)
            z_axis = np.array(f['depth_mm']) if 'depth_mm' in f else None

        # Replace file NoData with NaN if present
        nod = a.get('NoDataValue', None)
        if nod is not None:
            try:
                nod = float(nod)
                values = np.where(np.isclose(values, nod), np.nan, values) if np.ndim(values) else (np.nan if np.isclose(values, nod) else values)
            except Exception:
                pass

        return values, z_axis, crs


def oliktok_point(sensor='s1', rmethod='hadamard', resolution='40m', fnout=None, overwrite=True):
    # thresh = 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    # xticks_im = (25, 65, 105, 145)
    # yticks_im = (25, 65, 105)
    # ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]
    # ys = [(0.05, 0.25), (0.25, 0.45)]

    # profile = ((-149.8916,70.4955), (-149.8867,70.4967))
    # profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))
    # profile = ((392260, 7823681), (392511,7823824))
    # xy_ref = np.array([-149.8504,70.4836])[:, np.newaxis]
    # resolution = '40m'
    res0 = read_results(path_results(sensor=sensor, year=years[0], rmethod=rmethod, resolution=resolution), overwrite=overwrite)
    res1 = read_results(path_results(sensor=sensor, year=years[1], rmethod=rmethod, resolution=resolution), overwrite=overwrite)
    print(res0['e_mean'].shape)
    geospatial = res0['geospatial']
    shape = (res0['e_mean'].shape[0], res0['e_mean'].shape[1])
    print(geospatial)
    gsp =Geospatial(geospatial.transform, geospatial.crs, shape=shape)
    xy_ref = np.array([7822273, 392278])
    a = get_rowcol(geospatial.transform, xy_ref)
    print(a)
    raise
    xy_ref = np.array([7822273, 392278])[:, np.newaxis]
    pt = gsp.rowcol(xy_ref, geospatial.crs)
    print(f'rows and columns are :{pt}' )
    raise


    # assert res1['geospatial'] == geospatial

    # fig, axs = prepare_figure(ncols=2, nrows=3, sharex='none', sharey='none')
    initialize_matplotlib()
    fig = plt.figure(figsize=(14, 5.2))
    ax.plot(date_list, defo_median, marker='o', markersize=8, markerfacecolor=color, markeredgecolor='black',
            linestyle='-', color='black', label=legend_label, linewidth=2)
    # start_date = mdates.datestr2num('12/31/2016')
    # end_date = mdates.datestr2num('12/31/2022')
    # ax.set_xlim([start_date, end_date])
    # ax = set_x_date(ax, nmonths=12)

    y_ticks = np.arange(ylim[0] + 1, 0.1, step=4)
    ax.set_ylim(ylim)
    ax.set_yticks(y_ticks)
    # ax.set_xlabel('Years')
    ax.set_ylabel('Interannual deformation ($\hat{d}_{e}$)')
    # fig.set_size_inches((3.8, 3), forward=True)
    # gs = gridspec.GridSpec(
    #     2, 8, left=0.007, right=0.85, top=0.95, bottom=0.1, wspace=0.1, hspace=0.4)
    # axs = [
    #     [fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:8])],
    #     [fig.add_subplot(gs[1, 0:4]), fig.add_subplot(gs[1, 4:8])],
    #     # [fig.add_subplot(gs[2, 1:7]), fig.add_subplot(gs[2, 7:8])]
    # ]
    # axs[-1][-1].set_axis_off()
    # labels = [
    #     f'a) 2023: excess ice {str(int(ys[0][0]*100))}--{str(int(ys[0][1]*100))} cm',
    #     f'b) 2023: excess ice {str(int(ys[1][0]*100))}--{str(int(ys[1][1]*100))} cm',
    #     f'c) 2024: excess ice {str(int(ys[0][0]*100))}--{str(int(ys[0][1]*100))} cm',
    #     f'd) 2024: excess ice {str(int(ys[1][0]*100))}--{str(int(ys[1][1]*100))} cm']
    #
    # for jyear, res in enumerate([res0, res1]):
    #     # K, geospatial_K = read_K(fnK)
    #     # invalid = invalid_mask(
    #     #     K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)
    #     for jy, y in enumerate(ys):
    #         print(jy)
    #         jy0, jy1 = _get_index(res['ygrid'], y[0]), _get_index(res['ygrid'], y[1])
    #         _e_mean = np.mean(res['e_mean'][..., jy0:jy1], axis=-1)
    #         # _e_mean[invalid] = np.nan
    #         # fnout_emean = os.path.join(fnout_dir, f'emean_{years[jyear]}_{int(y[0]*100)}-{int(y[1]*100)}.tif')
    #         fnout_emean = fnout_dir / f'emean_{years[jyear]}_{int(y[0] * 100)}-{int(y[1] * 100)}.tif'
    #         save_geotiff(np.expand_dims(_e_mean, axis=0), geospatial, fnout=fnout_emean)
    #         ax = axs[jyear][jy]
    #         # crop_edges = True
    #         # if crop_edges:
    #         #     row_start, row_end, col_start, col_end = crop_nan_edges(_e_mean)
    #         #     print(_e_mean.shape)
    #         #     print(row_start, row_end, col_start, col_end)
    #         #     raise
    #         #     squeezed_emean = _e_mean[row_start:row_end + 1, col_start:col_end + 1]
    #         #     _e_mean = squeezed_emean
    #         im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
    #         ax.set_facecolor('#aaaaaa')
    #         # ax.set_xticks(xticks_im)
    #         # ax.set_yticks(yticks_im)
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    #         ax.grid(color='#dddddd', linewidth=0.4, alpha=0.5)
    # add_scalebar(axs[0][0], geospatial.upscaled(upscale), length=1000, label='1 km')
    # # optical = res0['optical'][::-1, ...][0:3]
    # # ax = axs[-1][0]
    # # ax.imshow(contrast(np.moveaxis(optical, 0, -1), percentiles=(2.0, 92.0)))
    # # ax.contour(
    # #     res0['dem'][0, ...], colors=['#ffffff'], linewidths=0.4, alpha=0.4, levels=10)
    # # rc_ref = np.array(geospatial.upscaled(upscale).rowcol(xy_ref))[:, 0]
    # # ax.plot(
    # #     rc_ref[1], rc_ref[0], linestyle='none', ms=2.5, marker='x',
    # #     mec=colslist[0], mfc=colslist[0], zorder=9)
    # # ax.set_xticks(np.array(xticks_im) * upscale)
    # # ax.set_yticks(np.array(yticks_im) * upscale)
    # # ax.set_xticklabels([])
    # # ax.set_yticklabels([])
    # # ax.grid(color='#aaaaaa', linewidth=0.4)
    # pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])
    #
    # # rc = pi._rowcol_endpoints
    # # label = f'T1'
    # # add_arrow_line(
    # #     ax, rc, label=label, c='#ffffff', lw=0.7, alpha=0.9, dlabel=(380, -180), hwidth=140,
    # #     hlength=180)
    # # _xy_site = geospatial.upscaled(upscale).rowcol(site)[:, 0]
    # # ax.plot(
    # #     _xy_site[1], _xy_site[0], c='#ffffff', linestyle='none',
    # #     marker='o', ms=5, mfc='none')
    # # ax.text(_xy_site[1] + 120, _xy_site[0] - 160, 'HV', c='#ffffff')
    #
    # # ax.text(0.01, -0.06, 'Planet Labs', ha='left', va='top', transform=ax.transAxes)
    #
    # xticks = [0, 100, 200]
    # yticks = (0.0, 0.1, 0.2, 0.3, 0.4)
    # ymax = 0.45
    #
    # # plabels = [
    # #     (0.13, 'inactive fp'), (0.45, 'abandoned fp'), (0.75, 'rocky'), (0.93, 'slope')]
    # x_ylabel = -0.12
    #
    # # plot_profile(
    # #     axs[2][0], res0['e_mean'], geospatial, profile, im_frac=res0['frac_thawed'],
    # #     ymax=ymax, vlim=elim, ygrid=res0['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
    # #     labels=None, x_ylabel=x_ylabel)
    # # axs[2][0].text(
    # #     0.05, 0.19, '$y_\\mathrm{f}$', c='#ffffff', transform=axs[-1][1].transAxes, alpha=0.6)
    # # plot_profile(
    # #     axs[-1][2], res1['e_mean'], geospatial, profile, im_frac=res1['frac_thawed'],
    # #     ymax=ymax, vlim=elim, ygrid=res1['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
    # #     labels=plabels, x_ylabel=x_ylabel)
    # #
    # cax = axs[0][-1].inset_axes([1.05, 0.1, 0.15, 0.80])
    # cax.text(1.0, 1.16, '$e$ [-]', ha='center', va='baseline', transform=cax.transAxes)
    # plt.colorbar(im_e, cax, shrink=0.5, orientation='vertical', ticks=[0.0, 0.25, 0.50])

    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.010, 1.035, lab, ha='left', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout, dpi=300)
        plt.show()

def oliktok_profile(years, sensor='s1', rmethod='hadamard', wavelength=0.055, fnout=None, overwrite=True):
    import rasterio
    from rasterio.plot import show as rio_show
    optical_tif = '/export/data/Experiments/geodata/oliktok/sentinel2_false.tif'
    thresh = 4.3e-3
    upscale = 32
    cmap = cmap_e
    elim = (0.0, 0.5)
    ys = [(0.05, 0.25), (0.25, 0.40)]
    profile = ((-149.8916, 70.4955), (-149.8867, 70.4967))

    res0 = read_results(path_results(sensor=sensor, year=years[0], rmethod=rmethod), overwrite=overwrite)
    res1 = read_results(path_results(sensor=sensor, year=years[1], rmethod=rmethod), overwrite=overwrite)
    geospatial = res0['geospatial']

    # === Load optical image ===
    with rasterio.open(optical_tif) as src:
        red, green, blue = src.read(1), src.read(2), src.read(3)
        transform = src.transform
        rgb_image = process_rgb_image(red, green, blue, normalize=True, alpha=1.1)

    # === Setup figure ===
    initialize_matplotlib()
    fig = plt.figure(figsize=(5.2, 8.5))
    gs = gridspec.GridSpec(3, 8, left=0.03, right=0.98, top=0.96, bottom=0.08, hspace=0.38)

    ax_optical = fig.add_subplot(gs[0, :])
    ax_2023 = fig.add_subplot(gs[1, :])
    ax_2024 = fig.add_subplot(gs[2, :])

    # === Row 1: Optical image ===
    rio_show(rgb_image * 3, ax=ax_optical, transform=transform)
    ax_optical.set_title('a) Optical Image', loc='left')
    ax_optical.set_xticks([])
    ax_optical.set_yticks([])
    ax_optical.grid(False)

    # === Row 2: Profile 2023 ===
    plot_profile(
        ax_2023, res0['e_mean'], geospatial, profile, im_frac=res0['frac_thawed'],
        ymax=0.45, vlim=elim, ygrid=res0['ygrid'], cmap=cmap, xticks=[0, 100, 200],
        yticks=(0.0, 0.1, 0.2, 0.3, 0.4), x_ylabel=-0.12)
    ax_2023.set_title('b) 2023 Profile', loc='left')

    # === Row 3: Profile 2024 ===
    plot_profile(
        ax_2024, res1['e_mean'], geospatial, profile, im_frac=res1['frac_thawed'],
        ymax=0.45, vlim=elim, ygrid=res1['ygrid'], cmap=cmap, xticks=[0, 100, 200],
        yticks=(0.0, 0.1, 0.2, 0.3, 0.4), x_ylabel=-0.12)
    ax_2024.set_title('c) 2024 Profile', loc='left')

    # Add scalebar to first profile row
    add_scalebar(ax_2023, geospatial.upscaled(upscale), length=1000, label='1 km')

    # Optional: Colorbar for both profiles
    im_sample = ax_2023.images[0]
    cax = fig.add_axes([0.85, 0.58, 0.015, 0.28])
    plt.colorbar(im_sample, cax=cax, ticks=[0.0, 0.25, 0.5])
    cax.text(1.1, 1.02, '$e$ [-]', ha='center', va='bottom', transform=cax.transAxes)

    if fnout:
        plt.savefig(fnout, dpi=300)
    else:
        plt.show()

def oliktok_map_alos2(year, sensor='alos2', rmethod='mintpy', wavelength=0.055, fnout=None, overwrite=True):
    thresh = 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    # xticks_im = (25, 65, 105, 145)
    # yticks_im = (25, 65, 105)
    # ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]
    ys = [(0.05, 0.25), (0.25, 0.45)]

    # profile = ((-149.8916,70.4955), (-149.8867,70.4967))
    # profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))
    # profile = ((392260, 7823681), (392511,7823824))
    # xy_ref = np.array([-149.8504,70.4836])[:, np.newaxis]

    res = read_results(path_results(sensor=sensor, year=year, rmethod=rmethod), overwrite=overwrite)
    # res1 = read_results(path_results(sensor=sensor, year=years[1], rmethod=rmethod), overwrite=overwrite)
    geospatial = res['geospatial']
    # geospatial.CRS = 'EPSG:4326'

    # assert res1['geospatial'] == geospatial

    # fig, axs = prepare_figure(ncols=2, nrows=3, sharex='none', sharey='none')
    initialize_matplotlib()
    fig = plt.figure()
    fig.set_size_inches((4.50, 2), forward=True)
    gs = gridspec.GridSpec(
        1, 8, left=0.007, right=0.85, top=0.95, bottom=0.1, wspace=0.2, hspace=0.4)
    axs = [
        [fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:8])],
        # [fig.add_subplot(gs[1, 1:7]), fig.add_subplot(gs[1, 7:8])]
    ]
    # axs[-1][-1].set_axis_off()
    labels = [
        f'a) 2024: excess ice {str(int(ys[0][0]*100))}--{str(int(ys[0][1]*100))} cm',
        f'b) 2024: excess ice {str(int(ys[1][0]*100))}--{str(int(ys[1][1]*100))} cm']

    # for jyear, res in enumerate([res0, res1]):
        # K, geospatial_K = read_K(fnK)
        # invalid = invalid_mask(
        #     K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)
    for jy, y in enumerate(ys):
        jy0, jy1 = _get_index(res['ygrid'], y[0]), _get_index(res['ygrid'], y[1])
        _e_mean = np.mean(res['e_mean'][..., jy0:jy1], axis=-1)
        # _e_mean[invalid] = np.nan
        fnout_emean = fnout_dir / f'emean_{year}_{int(y[0] * 100)}-{int(y[1] * 100)}_alos2.tif'
        save_geotiff(np.expand_dims(_e_mean, axis=0), geospatial, fnout=fnout_emean)
        # fnout_emean = os.path.join(fnout_dir, f'emean_{years}_{int(y[0]*100)}-{int(y[1]*100)}.tif')
        # save_geotiff(np.expand_dims(_e_mean, axis=0), geospatial, fnout=fnout_emean)
        ax = axs[0][jy]
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
    # add_scalebar(axs[0][0], geospatial.upscaled(upscale), length=1000, label='1 km')
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

    # xticks = [0, 250, 500, 750, 1000]
    # yticks = (0.0, 0.1, 0.2, 0.3, 0.4)
    # ymax = 0.45

    # plabels = [
    #     (0.13, 'inactive fp'), (0.45, 'abandoned fp'), (0.75, 'rocky'), (0.93, 'slope')]
    # x_ylabel = -0.12

    # plot_profile(
    #     axs[1][0], res['e_mean'], geospatial, profile, im_frac=res['frac_thawed'],
    #     ymax=ymax, vlim=elim, ygrid=res['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
    #     labels=None, x_ylabel=x_ylabel)
    # axs[2][0].text(
    #     0.05, 0.19, '$y_\\mathrm{f}$', c='#ffffff', transform=axs[-1][1].transAxes, alpha=0.6)
    # plot_profile(
    #     axs[-1][2], res1['e_mean'], geospatial, profile, im_frac=res1['frac_thawed'],
    #     ymax=ymax, vlim=elim, ygrid=res1['ygrid'], cmap=cmap, xticks=xticks, yticks=yticks,
    #     labels=plabels, x_ylabel=x_ylabel)
    #
    cax = axs[0][-1].inset_axes([1.05, 0.1, 0.15, 0.80])
    cax.text(1.0, 1.16, '$e$ [-]', ha='center', va='baseline', transform=cax.transAxes)
    plt.colorbar(im_e, cax, shrink=0.5, orientation='vertical', ticks=[0.0, 0.25, 0.50])

    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.010, 1.035, lab, ha='left', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout, dpi=300)
        plt.show()
def oliktok_map_profiles(years, sensor='s1', rmethod='hadamard', wavelength=0.055, fnout=None, overwrite=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scripts.plotting import (
        prepare_figure, cmap_e, colslist, _get_index, contrast, initialize_matplotlib,
        add_scalebar, plot_profile, add_arrow_line, ProfileInterpolator)
    thresh = 4.3e-3
    upscale = 32

    cmap = cmap_e
    elim = (0.0, 0.5)
    xticks_im = (25, 65, 105, 145)
    yticks_im = (25, 65, 105)
    # ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]
    ys = [(0.05, 0.25), (0.25, 0.40)]

    # profile = ((-148.8013, 69.1609), (-148.7717, 69.1609))
    # profile = ((-148.7950, 69.1466), (-148.7655, 69.1466))
    profile = ((392260, 7823681), (392511,7823824))
    # xy_ref = np.array([-149.8504,70.4836])[:, np.newaxis]

    # res0 = read_results(
    #     path_results(years[0]), fnimraw=fnimraw, fndemraw=fndemraw, upscale=upscale,
    #     overwrite=overwrite)
    # res0 = read_results(path_results(years[0]), upscale=upscale, overwrite=overwrite)
    # print(path_results(years))
    res0 = read_results(path_results(sensor=sensor, year=years[0], rmethod=rmethod), overwrite=overwrite)
    res1 = read_results(path_results(sensor=sensor, year=years[1], rmethod=rmethod), overwrite=overwrite)
    # print(res0)
    # print(res0['e_mean'][:, :, 350])
    # print(res1['e_mean'][:, :, 150])
    geospatial = res0['geospatial']
    # assert res1['geospatial'] == geospatial

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
        # fnK = f'/export/data/Experiments/stacks/OliktokPoint_P102D/{years[jyear]}/proc/hadamard/geocoded/K_vec.geo.tif'
        # fnK = f'/export/data/Experiments/stacks/OliktokPoint_ALOS2/mintpy_outputs/2024/defo_history_cov.tif'
        # K, geospatial_K = read_K(fnK)
        # invalid = invalid_mask(
        #     K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)
        for jy, y in enumerate(ys):
            jy0, jy1 = _get_index(res['ygrid'], y[0]), _get_index(res['ygrid'], y[1])
            _e_mean = np.mean(res['e_mean'][..., jy0:jy1], axis=-1)
            # _e_mean[invalid] = np.nan
            fnout_emean = os.path.join(fnout_dir, f'emean_{years}_{int(y[0]*100)}-{int(y[1]*100)}.tif')
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
    pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])
    print(pi)
    rc = pi._rowcol_endpoints
    print(rc)
    label = f'T1'
    add_arrow_line(
        ax, rc, label=label, c='#ffffff', lw=0.7, alpha=0.9, dlabel=(380, -180), hwidth=140,
        hlength=180)
    _xy_site = geospatial.upscaled(upscale).rowcol(site)[:, 0]
    ax.plot(
        _xy_site[1], _xy_site[0], c='#ffffff', linestyle='none',
        marker='o', ms=5, mfc='none')
    # ax.text(_xy_site[1] + 120, _xy_site[0] - 160, 'HV', c='#ffffff')
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

def check_results(year, sensor='s1', rmethod='hadamard', wavelength=0.055, overwrite=True):
    res0 = read_results(path_results(sensor=sensor, year=year, rmethod=rmethod), overwrite=overwrite)
    print(res0.keys())
    print(res0['ygrid'].shape, res0['geospatial'])
    print(res0['e_mean'].shape, res0['frac_thawed'].shape)

if __name__ == '__main__':
    # from scripts.pathnames import paths
    years = (2023, 2024)
    # site_name = 'Utqiagvik'
    site_name = 'oliktok'
    sensor = 's1'
    # sensor = 'alos2'
    resolution = '40m'
    # resolution = '80m'
    # stack_method = 'hadamard'
    stack_method = 'mintpy'
    fnout_dir = Path('/export/data/Experiments/gie/processed/oliktok/postproc/')
    fig_dir = f'/home/jchen20/Dropbox/figures/stacks/{site_name}/gie_{site_name}_{sensor}_{stack_method}_{resolution}.png'
    # xy_ref = np.array([7822273, 392278])
    # emean_h5 = '/export/data/Experiments/gie/processed/oliktok/s1/2024/mintpy/40m/e_mean.h5'
    # vals, depth_mm, crs = get_point_h5(emean_h5, 392492,7823893)

    # check_results(years[0], sensor='s1', rmethod='hadamard', wavelength=0.055, overwrite=True)
    oliktok_map(years=years, sensor=sensor, rmethod=stack_method, resolution=resolution, fnout=fig_dir, overwrite=False)
    # oliktok_point(sensor=sensor, rmethod=stack_method, resolution=resolution, fnout=fig_dir)

    # sensor = 'alos2'
    # oliktok_map_alos2(years[1], sensor=sensor, rmethod=stack_method, wavelength=0.236, fnout=fig_dir, overwrite=False)
    exit()

    # oliktok_profile(years=years, sensor=sensor, rmethod=stack_method, wavelength=0.055, fnout=fig_dir, overwrite=False)
