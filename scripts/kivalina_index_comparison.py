'''
Created on Jun 28, 2023

@author: simon
'''

import os
import shapely
import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from analysis import Geospatial, load_object, read_geotiff, K_from_K_vec, save_object, InversionResults

path0 = '/home/simon/Work/gie/processed/kivalina/index/'
pathfig = '/home/simon/Work/gie/figures/index/'

# add geospatial with custom AEA (QGIS)
crs = CRS.from_epsg(3572)
posting = 50
transform = Affine(posting, 0.0, -615854, 0.0, -posting, -2381365)
shape = (124, 124)
geospatial_subset = Geospatial(transform, crs, shape=shape)

pathm1 = os.path.abspath(os.path.join(path0, os.pardir))
_fnK = os.path.join(pathm1, f'2019_index', 'K_vec.geo.tif')
geospatial_native = Geospatial.from_file(os.path.join(path0, _fnK))

def resample_iceoptical(fniceoptical, geospatial):
    iceoptical = gpd.read_file(fniceoptical).to_crs(geospatial.crs)
    iceoptical = iceoptical[iceoptical['include'] == 1]
    return geospatial.rasterize(iceoptical, field='code')
    # geom = [(shps, vals) for shps, vals in zip(iceoptical.geometry, iceoptical['code'])]
    # rasterized = features.rasterize(
    #     geom, out_shape=geospatial.shape, fill=-1, out=None,
    #     transform=geospatial.transform, default_value=-1, dtype=np.int64)[np.newaxis, ...]
    # return rasterized

def resample_scenario(path0, scenario, geospatial, metrics=('mean', 'var'), apply_mask=True):
    geospatial_mean = load_object(os.path.join(path0, scenario, 'ir.p'))['geospatial']
    def _read(metric, mask=None):
        fnm = os.path.join(path0, scenario, f'e_mean_period_{metric}.npy')
        metric = np.moveaxis(load_object(fnm), -1, 0)
        if mask is not None:
            np.putmask(metric, np.broadcast_to(mask, metric.shape), np.nan)
        arr, _ = geospatial.warp(metric, geospatial_mean)
        return arr
    dictout = {}
    year = scenario[:4]
    if apply_mask:
        mask = mask_year(path0, year)
    for metric in metrics:
        arr = _read(metric, mask=mask)
        dictout[metric] = arr
    return dictout

def mask_year(path0, year, thresh=1.5, opening=1, closing=1, geospatial_out=None):
    pathm1 = os.path.abspath(os.path.join(path0, os.pardir))
    fn = os.path.join(pathm1, f'{year}_index', 'K_vec.geo.tif')
    if geospatial_out is None:
        K_vec = read_geotiff(fn)
        geospatial_out = Geospatial.from_file(fn)
    else:
        K_vec = geospatial_out.warp_from_file()
    K = K_from_K_vec(K_vec)
    mask = K[-1, -1, ...] > thresh
    if opening is not None:
        from scipy.ndimage import binary_opening
        mask = binary_opening(mask, iterations=opening)
    if closing is not None:
        from scipy.ndimage import binary_closing
        mask = binary_closing(mask, iterations=closing)
    return mask

def _read_config(config, indranges_names, geospatial, path0):
    _ind = indranges_names.index(config[1])
    em = resample_scenario(path0, config[0], geospatial)
    return {metric: em[metric][_ind, ...] for metric in em}

def plot_subset(configs, indranges_names, path0, config_labels=None, fntmp=None, overwrite=False):
    metrics = ['mean', 'var']
    def _2d_kde(x, y, lim=(0, 0.7), steps=100):
        import scipy.stats as st
        xx, yy = np.mgrid[lim[0]:lim[1]:1j * steps, lim[0]:lim[1]:1j * steps]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        valid = np.logical_and(np.isfinite(x), np.isfinite(y))
        values = np.vstack([x[valid], y[valid]])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        return f
    def _prepare_data():
        if fntmp is None or not os.path.exists(fntmp) or overwrite:
            em = [_read_config(config, indranges_names, geospatial_subset, path0) for config in configs]
            kd = []
            for jconfig, config in enumerate(configs[:-1]):
                kddict = {}
                for metric in metrics:
                    x, y = em[-1][metric], em[jconfig][metric]
                    if metric == 'var': x, y = np.sqrt(x), np.sqrt(y)
                    kddict[metric] = _2d_kde(x.flatten(), y.flatten(), lim=lims[metric])
                kd.append(kddict)
            if fntmp is not None:
                save_object((em, kd), fntmp)
        else:
            em, kd = load_object(fntmp)
        return em, kd        
    from scripts.plotting import initialize_matplotlib, cmap_e, colslist, add_scalebar, prepare_figure
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle    
    import copy
    fig, axs = prepare_figure(
        nrows=3, ncols=len(configs), figsize=(1.8, 1.2), remove_spines=False, sharex='none', sharey='none',
        top=0.970, bottom=0.070, left=0.080, right=0.845, wspace=0.250, hspace=0.050)
    lims = {'mean': (0.0, 0.7), 'var': (0.00, 0.20)}
    e_ticks = {'mean': (0.0, 0.3, 0.6), 'var': (0.00, 0.10, 0.20)}
    cmap = copy.copy(cmap_e)
    cmap.set_bad(color='#444444')
    e_lim = (0.00, 0.70)
    
    em, kd = _prepare_data()
    for jconfig, config in enumerate(configs):
        axs[0, jconfig].imshow(em[jconfig]['mean'], cmap=cmap, vmin=e_lim[0], vmax=e_lim[1])
        if jconfig < len(configs) - 1:
            for jmetric, metric in enumerate(metrics):
                ax, lim = axs[jmetric + 1, jconfig], lims[metric]
                ax.imshow(
                    kd[jconfig][metric].T, origin='lower', extent=lim * 2, cmap=cmap)
                ax.set_xticks(e_ticks[metric])
                ax.set_yticks(e_ticks[metric])
                ax.plot(lim, lim, lw=0.5, c='#dddddd', alpha=0.3)
    
    cols_iceo = ['#666666', colslist[0], colslist[1], colslist[2]]
    lcm = ListedColormap(cols_iceo)
    iceo = resample_iceoptical(fniceoptical, geospatial_subset)[0, ...]
    axs[1, -1].imshow(iceo, cmap=lcm, interpolation='nearest')

    cax_extent = [1.06, 0.13, 0.10, 0.60]
    height, vpos_label = 0.6, 1.21
    cax = axs[1, -1].inset_axes(cax_extent)
    cax_labels = ['ice poor', 'ice rich', 'indeterminate']
    for jcol, col in enumerate(cols_iceo[1:]):
        cax.add_patch(Rectangle((0, -jcol), 1.0, -height, color=col))
        cax.text(2.0, -(jcol + 0.4), cax_labels[jcol], ha='left', va='center', transform=cax.transData)
    cax.set_ylim((-(len(cols_iceo) - 2 + height), 0))
    cax.set_xlim((0, 1))
    cax.axis('off')
    cax.text(0.00, vpos_label, 'independent map', ha='left', va='baseline', transform=cax.transAxes)
    from matplotlib import cm
    from matplotlib.colors import Normalize
    cax0 = axs[0, -1].inset_axes(cax_extent)
    cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(*e_lim, clip=True), cmap=cmap), cax=cax0)
    cbar.set_ticks([e_lim[0], e_lim[1] / 2, e_lim[1]])
    cbarlabel = '$\\bar{e}$ [$-$]'
    cax0.text(1.50, vpos_label, cbarlabel, ha='center', va='baseline', transform=cax0.transAxes)

    # scale bar
    add_scalebar(axs[-1, -1], geospatial_subset, length=2e3, y=0.25, dx=-0.68, label='2 km', ylab=0.18)
    axs[-1, -1].text(
        1.03, 0.5, 'Sentinel-2 true-color', rotation=270, ha='left', va='center',
        transform=axs[-1, -1].transAxes)
    
    fnS2 = '/home/simon/Work/Kivalina/optical/Sentinel2/20190711_rgb.vrt'
    S2, _ = geospatial_subset.warp_from_file(fnS2)
    S2 = S2[::-1,:,:]
    def _normalize(im):
        anc = np.nanpercentile(im, (2, 99), axis=(1, 2))
        im -= anc[0,:, np.newaxis, np.newaxis]
        im /= (anc[1,:] - anc[0,:])[:, np.newaxis, np.newaxis]
        return np.moveaxis(im, 0, -1)
    axs[2, -1].imshow(_normalize(S2))

    
    # labels
    if config_labels is not None:
        for jax, ax in enumerate(axs[0,:]):
            ax.text(0.50, 1.11, config_labels[jax], ha='center', va='baseline', transform=ax.transAxes)
    for ax in axs[-1, :-1]:
        ax.text(
            0.50, -0.33, 'reference $\\mathrm{std}(\\bar{e})$ [$-$]', ha='center', va='baseline', 
            transform=ax.transAxes)
    ylab = ('$\\bar{e}$ [$-$]', '$\\mathrm{std}(\\bar{e})$ [$-$]')
    for jax, ax in enumerate(axs[1:, 0]):
        ax.text(-0.33, 0.50, ylab[jax], rotation=90, ha='right', va='center', transform=ax.transAxes)
    # lines
    import matplotlib.lines as mlines
    y_line = 0.96
    x_lines = [(axs[0, 0].get_position().x0, axs[0, -2].get_position().x1),
               (axs[0, -1].get_position().x0, axs[0, -1].get_position().x1)]
    for x_line in x_lines:
        line = mlines.Line2D(x_line, [y_line, y_line], transform=fig.transFigure, c='#666666', lw=0.5)
        fig.lines.extend([line])
    for ax in np.concatenate((axs[0, :], axs[:, -1])):
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    for ax in axs[1:, 1:-1].flatten():
        ax.tick_params(labelleft=False)
    import matplotlib.pyplot as plt
    plt.show()

def rasterize_cores(fncores, geospatial):
    # breaks down when multiple coords map to the same pixel
    cores = gpd.read_file(fncores).to_crs(geospatial.crs)
    imap_ = geospatial.rasterize(cores, field='code')
    mask_ = geospatial.rasterize(cores, field='include')
    imap_[mask_ == 0] = 255
    return imap_

def violin_plot(config, fncores=None):
    from scripts.plotting import colslist, prepare_figure
    imap = resample_iceoptical(fniceoptical, geospatial_subset)[0, ...]
    emean = _read_config(config, indranges_names, geospatial_subset, path0)['mean']
    labels = ['ice poor', 'ice rich', 'indeterminate']
    rs = np.random.RandomState(seed=1)

    fig, ax = prepare_figure(
        nrows=1, ncols=1, figsize=(1, 0.6), remove_spines=False, top=0.80, bottom=0.04, left=0.22,
        right=0.98)
    ypos = []
    offs = 0.3
    offsc = -0.5
    alphas = [1.0, 1.0, 0.5]
    col = colslist[0]
    if fncores is not None:
        cores = rasterize_cores(fncores, geospatial_native)[0, ...]
        emean_cores = _read_config(config, indranges_names, geospatial_native, path0)['mean']

    for jcode, code in enumerate([0, 1, 2]):
        emean_code = emean[np.logical_and(imap == code, np.isfinite(emean))].flatten()
        ypos_ = -jcode - 2 * offs * (code == 2)
        ypos.append(ypos_)
        bp = ax.violinplot(
            emean_code, vert=False, positions=(ypos_,), showextrema=False, widths=0.7)
        if fncores is not None:
            emean_code = emean_cores[np.logical_and(cores == code, np.isfinite(emean_cores))]
            ypos_cores = ypos_ + offsc + rs.uniform(0.0, 0.1, size=emean_code.shape)
            ax.plot(
                    emean_code, ypos_cores, linestyle='none', marker='o', mfc='none',
                    mew=0.6, mec=col, ms=4)
        if jcode == 1:
            xpos = 0.75
            ax.text(xpos, ypos_, 'map', ha='right', va='center',
                    bbox=dict(facecolor='#ffffff', edgecolor='none'))
            if fncores is not None:
                ax.text(xpos, ypos_ + offsc, 'cores', ha='right', va='center')
        bp['bodies'][0].set_facecolor(col)
        bp['bodies'][0].set_alpha(alphas[jcode])
        bp['bodies'][0].set_edgecolor('none')

        ax.xaxis.tick_top()
        ax.text(
            0.5, 1.18, '$\\bar{e}$ [$-$]', transform=ax.transAxes,
            ha='center')

    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim((ypos[-1] - 0.4, ypos[0] + 0.5))
    import matplotlib.pyplot as plt
    plt.show()

def plot_profile_time_series(path0, scenario='2019r'):
    from scripts.plotting import (
        initialize_matplotlib, cmap_e, colslist, _get_index, ProfileInterpolator,
        contrast, add_arrow_line, plot_profile, add_scalebar)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    pathres = os.path.join(path0, scenario)
    profiles = [((-164.3881, 67.8120), (-164.4469, 67.8272))]

    fig = plt.figure()
    initialize_matplotlib()
    fig.set_size_inches((7.08, 2.68), forward=True)
    gs = gridspec.GridSpec(
        8, 2, left=0.005, right=0.995, top=0.980, bottom=0.086, wspace=4.50, hspace=0.03)
    axs = [[plt.subplot(gs[0:3, 0:1])]
           [plt.subplot(gs[4:4, 0:1])],
           [plt.subplot(gs[5:7, 0:0]), plt.subplot(gs[5:7, 1:1])]]

    ir = InversionResults.from_file(os.path.join(pathres, 'ir.p'))
    geospatial = ir.geospatial
    ygrid = ir.ygrid
    save_object(geospatial, os.path.join(pathres, 'geospatial.p'))
    save_object(ygrid, os.path.join(pathres, 'ygrid.p'))

    e_mean = np.load(os.path.join(pathres, 'e_mean.npy'))
    frac_thawed = np.load(os.path.join(pathres, 'frac_thawed_None.npy'))
    cmap = cmap_e
    elim = (0.0, 0.5)
    xticks_im = (25, 65, 105, 145)
    yticks_im = (10, 50, 90)
    geospatial = geospatial_native
    for jp, profile in enumerate(profiles):
        pi = ProfileInterpolator(geospatial, profile[0], profile[1])
        rc = pi._rowcol_endpoints

    # read e_index

    # load unw and show two time series

if __name__ == '__main__':
    fnsubset = os.path.join(path0, 'subset.gpkg')
    fniceoptical = os.path.join(path0, 'iceoptical.gpkg')
    fncores = os.path.join(path0, 'cores2005.gpkg')
    ft = load_object(os.path.join(path0, '2019r', 'forcing_timing.p'))
    indranges_names = ft['indranges_names']
    configs = [
        ('2019', 'TDD900_lastday'), ('2019', 'TDD1000_lastday'), ('2018', 'TDD900_lastday'),
        ('2019r', 'TDD900_lastday')]
    config_labels = ['later scene', 'later start', 'cooler summer', 'reference']

    # violin_plot(configs[-1], fncores=fncores)
    fntmp = os.path.join(pathfig, 'kde.p')
    plot_subset(configs, indranges_names, path0, config_labels=config_labels, fntmp=fntmp)
    # plot_profile_time_series(path0, scenario='2019r')
