'''
Created on Sep 19, 2022

@author: simon
'''
import os
import numpy as np
from analysis import Geospatial, save_geotiff, save_object, load_object, InversionResults

def invalid_mask(K, thresh, geospatial_K, geospatial, ind1=0, ind2=-1, wavelength=0.055):
    from scipy.ndimage import binary_dilation, binary_opening, binary_closing
    from analysis import add_atmospheric
    K = add_atmospheric(K, 0.0, wavelength=wavelength)
    K_last = K[ind1, ind1, ...] + K[ind2, ind2, ...] - 2 * K[ind1, ind2, ...]
    K_last_crop, _ = geospatial.warp(K_last, geospatial_K)
    s1 = np.array(
        [[ 0, 1, 0], [ 1, 1, 1], [0, 1, 0]])
    s2 = np.array(
        [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])

    # invalid = binary_opening(
    #     binary_dilation(K_last_crop > thresh ** 2, s1), s2, border_value=1)
    invalid = binary_dilation(binary_opening(binary_closing(K_last_crop > thresh ** 2, s1), s1), s1)
    return invalid

def read_core_data(fngpkg, geospatial, layer=None):
    import fiona
    import geopandas as gpd
    if layer is None: layer = fiona.listlayers(fngpkg)[0]
    gdf = gpd.read_file(fngpkg, layer=layer).to_crs(geospatial.crs)
    rcs = []
    for index, row in gdf.iterrows():  # ugly
        rc = tuple(geospatial.rowcol(np.array(row.geometry.xy))[:, 0])
        rcs.append(rc)
    gdf['rowcol'] = rcs
    return gdf

def resample_dem(geospatial, fnraw, fnresampled, upscale=None, overwrite=False):
    from analysis import read_geotiff
    if os.path.exists(fnresampled) and not overwrite:
        return read_geotiff(fnresampled)
    else:
        dem, _gs = geospatial.warp_from_file(fnraw, upscale=upscale)
        save_geotiff(dem, _gs, fnresampled)
        return dem

def plot_kivalina(fnout=None):
    from analysis import read_K
    import matplotlib.gridspec as gridspec
    from scripts.plotting import (
        initialize_matplotlib, cmap_e, colslist, _get_index, ProfileInterpolator,
        contrast, add_arrow_line, plot_profile, add_scalebar)
    import matplotlib.pyplot as plt

    pathres = '/home/simon/Work/gie/processed/kivalina/2019/hadamard/inversion/'
    fnK = '/home/simon/Work/gie/processed/kivalina/2019/hadamard/K_vec.geo.tif'
    fndemraw = '/home/simon/Work/Kivalina/optical/DEM/ArcticDEM/53_19_2_1_2m_v3.0_reg_dem.tif'
    fndemres = os.path.join(pathres, 'DEM.tif')
    fnimraw = '/home/simon/Work/Kivalina/optical/Planet/Kivalina2019/20190625_220816_0e26/analytic_sr_udm2/20190625_220816_0e26_3B_AnalyticMS_SR.tif'
    fnimres = os.path.join(pathres, 'optical.tif')
    fngpkg = '/home/simon/Work/Kivalina/geology/cores2005.gpkg'
    upscale = 16
    wavelength, thresh = 0.055, 4.8e-3

    ir = InversionResults.from_file(os.path.join(pathres, 'ir.p'))
    geospatial = ir.geospatial
    ygrid = ir.ygrid
    save_object(geospatial, os.path.join(pathres, 'geospatial.p'))
    # geospatial = load_object(os.path.join(pathres, 'geospatial.p'))
    # ygrid = np.arange(0, 1.5, step=2e-3)

    K, geospatial_K = read_K(fnK)
    invalid = invalid_mask(K, thresh, geospatial_K, geospatial, ind1=4, wavelength=wavelength)
    profiles = [((-164.7440, 67.8420), (-164.7350, 67.8480)),
                ((-164.8171, 67.8500), (-164.8095, 67.8555))]
    plabels = [[(0.17, 'polygons'), (0.5, 'rocky bench'), (0.85, 'no polygons')],
               [(0.24, 'inactive floodplain'), (0.87, 'polygons')]]
    dem = resample_dem(geospatial, fndemraw, fndemres, upscale=upscale)
    optical = resample_dem(geospatial, fnimraw, fnimres, upscale=upscale)
    e_mean = np.load(os.path.join(pathres, 'e_mean.npy'))
    frac_thawed = np.load(os.path.join(pathres, 'frac_thawed_None.npy'))

    cores = read_core_data(fngpkg, geospatial)

    cmap = cmap_e
    elim = (0.0, 0.5)
    xticks_im = (25, 65, 105, 145)
    yticks_im = (10, 50, 90)
    ys = [(0.05, 0.15), (0.20, 0.30), (0.55, 0.65)]
    labels = [
        'a) excess ice 5--15 cm', 'b) excess ice 20--30 cm',
        'c) excess ice 55--65 cm', 'd) false-color image', 'e) transect T1',
        'f) transect T2']

    fig = plt.figure()
    initialize_matplotlib()
    fig.set_size_inches((7.08, 2.75), forward=True)
    gs = gridspec.GridSpec(
        2, 14, left=0.005, right=0.995, top=0.982, bottom=0.084, wspace=4.50, hspace=0.04)

    axs = [[plt.subplot(gs[0, 0:4]), plt.subplot(gs[0, 4:8]), plt.subplot(gs[0, 8:12])],
           [plt.subplot(gs[1, 0:4]), plt.subplot(gs[1, 4:9]), plt.subplot(gs[1, 9:])]]

    for jy, y in enumerate(ys):
        _e_mean = np.mean(
            e_mean[..., _get_index(ygrid, y[0]):_get_index(ygrid, y[1])], axis=-1)
        _e_mean[invalid] = np.nan
        ax = axs[0][jy]
        im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
        ax.set_facecolor('#aaaaaa')
        ax.set_xticks(xticks_im)
        ax.set_yticks(yticks_im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='#dddddd', linewidth=0.4)
    cols_scatter = [('#ffffff', '#cccccc'), ('#000000', '#333333')]
    marker_size, marker_lw, marker, alpha = 3.5, 0.8, 'o', 0.35
    c = [cols_scatter[int(x)] for x in cores.code]
    for _xy, _c in zip(cores.rowcol, c):
        ax.plot(
            _xy[1], _xy[0], linestyle='none', ms=marker_size,
            mec=_c[0], mew=marker_lw, mfc='none', marker=marker, zorder=10)
        ax.plot(
            _xy[1], _xy[0], linestyle='none', ms=marker_size,
            mec='none', mew=0.0, mfc=_c[1], marker=marker, zorder=9, alpha=alpha)
    ax.set_xlim(axs[0][0].get_xlim())
    ax.set_ylim(axs[0][0].get_ylim())
    optical = optical[::-1, ...][0:3]
    ax = axs[1][0]
    ax.imshow(contrast(np.moveaxis(optical, 0, -1)))
    ax.contour(dem[0, ...], colors=['#333333'], linewidths=0.4, alpha=0.4, levels=10)
    for jp, profile in enumerate(profiles):
        pi = ProfileInterpolator(geospatial.upscaled(upscale), profile[0], profile[1])
        rc = pi._rowcol_endpoints
        label = f'T{jp + 1}'
        add_arrow_line(
            ax, rc, label=label, c=colslist[0], lw=0.7, alpha=0.9, dlabel=(95, -15))
    ax.set_xticks(np.array(xticks_im) * upscale)
    ax.set_yticks(np.array(yticks_im) * upscale)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='#aaaaaa', linewidth=0.4)
    add_scalebar(ax, geospatial.upscaled(upscale), length=1000, label='1 km')

    xticks = [0, 200, 400, 600]
    yticks = (0.0, 0.2, 0.4, 0.6)

    ymax = 0.7  # 0.5
    plot_profile(
        axs[1][1], e_mean, geospatial, profiles[0], steps=512, ymax=ymax, vlim=elim,
        ygrid=ygrid, cmap=cmap, xticks=xticks, yticks=yticks, labels=plabels[0])
    plot_profile(
        axs[1][2], e_mean, geospatial, profiles[1], steps=512, ymax=ymax, vlim=elim,
        ygrid=ygrid, cmap=cmap, xticks=xticks, yticks=yticks, labels=plabels[1])
    
    bbox_r = axs[1][0].get_position()
    for ax in (axs[1][1], axs[1][2]):
        bbox_c = ax.get_position()
        ax.set_position(
            (bbox_c.x0, bbox_r.y0, bbox_c.x1 - bbox_c.x0, bbox_r.y1 - bbox_r.y0))

    lax = axs[0][-1].inset_axes([1.10, 0.58, 0.45, 0.40])
    lax.set_axis_off()
    x_scatter, y_scatter = [0.2, 0.8], 0.38
    labels_scatter = ['ice poor', 'ice rich']
    for _x, _c in zip(x_scatter, cols_scatter):
        lax.plot(
            _x, y_scatter, linestyle='none', ms=marker_size, transform=lax.transAxes,
            mec=_c[0], mew=marker_lw, mfc='none', marker=marker, zorder=10,)
        lax.plot(
            _x, y_scatter, linestyle='none', ms=marker_size, transform=lax.transAxes,
            mec='none', mew=0.0, mfc=_c[1], marker=marker, zorder=9, alpha=alpha)
    for _x, _l in zip(x_scatter, labels_scatter):
        lax.text(_x, 0.0, _l, va='baseline', ha='center', transform=lax.transAxes)
    lax.text(0.5, 0.9, 'upper permafrost', va='top', ha='center', transform=lax.transAxes)
    lax.plot(
        x_scatter[0], y_scatter, linestyle='none', ms=marker_size - 1, mew=0,
        transform=lax.transAxes, mfc='#666666', marker=marker, zorder=11)
    lax.plot(
        x_scatter[0], y_scatter, linestyle='none', ms=marker_size - 1.5, mew=0,
        transform=lax.transAxes, mfc='#ffffff', marker=marker, zorder=11)
    lax.plot(
        x_scatter[0], y_scatter, linestyle='none', ms=marker_size, mew=1.5,
        transform=lax.transAxes, mec='#666666', mfc='none', marker=marker, zorder=8)
    cax = axs[0][-1].inset_axes([1.10, 0.23, 0.45, 0.10])
    cax.text(0.5, 1.5, '$e$ [-]', ha='center', va='baseline', transform=cax.transAxes)
    plt.colorbar(im_e, cax, shrink=0.5, orientation='horizontal')
    for ax, lab in zip([ax for axr in axs for ax in axr], labels):
        ax.text(0.01, 1.04, lab, ha='left', va='baseline', transform=ax.transAxes)
    if fnout is not None:
        plt.savefig(fnout)
    else:
        plt.show()

if __name__ == '__main__':
    from scripts.pathnames import paths
    fnplot = os.path.join(paths['figures'], 'kivalina.pdf')
    plot_kivalina(fnplot)
    # e_mean_ = np.mean(e_mean[..., _get_index(ygrid, 0.40):_get_index(ygrid, 0.50)], axis=-1)  # 0.5
    # e_mean_[invalid] = nodata
    # print(np.nanpercentile(e_mean_, (10, 25, 50, 75, 90)))

    # profile_frac = profile_frac[:,:_get_index(ygrid, ymax)]

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(profile.T, vmin=0.0, vmax=vmax, cmap=cmap, alpha=1)  # profile_frac.T)
    # ax.set_yticks(_get_index(ygrid, yticks))
    # ax.set_yticklabels(yticks)
    # ax.set_xticks(_get_index(pi.distance_steps, xticks))
    # ax.set_xticklabels(xticks)
    # ax.set_xlabel('Distance [m]')
    # ax.set_ylabel('Depth [m]')
    # ax.set_ylims()
    # # plt.imshow(e_mean_)
    # plt.show()
    # save_geotiff(e_mean_[np.newaxis, ...], geospatial, os.path.join(pathres, 'e_mean.tif'), nodata=nodata)

