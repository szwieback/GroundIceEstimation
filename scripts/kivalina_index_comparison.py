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

from analysis import (
    Geospatial, read_geotiff_geospatial, load_object, read_geotiff, K_from_K_vec, save_object,
    InversionResultsMmap, assemble_tril)
from pip._vendor.webencodings import ascii_lower

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
        arrm = np.moveaxis(load_object(fnm), -1, 0)
        if mask is not None:
            np.putmask(arrm, np.broadcast_to(mask, arrm.shape), np.nan)
        if geospatial is not None:
            arrm, _ = geospatial.warp(arrm, geospatial_mean)
        return arrm
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
    from scripts.plotting import cmap_e, colslist, add_scalebar, prepare_figure
    from string import ascii_lowercase
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as path_effects
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
    for ax in axs[-1,:-1]:
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
    # ticks
    for ax in np.concatenate((axs[0,:], axs[:, -1])):
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    for ax in axs[1:, 1:-1].flatten():
        ax.tick_params(labelleft=False)
    # panels
    for jax, ax in enumerate(axs.flatten()):
        lab = f'{ascii_lowercase[jax]})'
        txt = ax.text(
            0.98, 0.04, lab, c='w', transform=ax.transAxes, ha='right', va='baseline')
        txt.set_path_effects(
            [path_effects.Stroke(linewidth=1.0, foreground='#111111'), path_effects.Normal()])
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
    ft = load_object(os.path.join(path0, config[0], 'forcing_timing.p'))
    indranges_names = ft['indranges_names']
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

def _read_timeseries_kivalina(year=2019, remove_last=False, overwrite=False):
    from scripts.kivalina_calibration import caldict
    from analysis import RationalQuadraticSepDiagCovMV, read_K, length_conversion, spatial_referencing
    from scripts.kivalina_index import kivalina_forcing, wavelength, geom

    path1 = os.path.join(pathm1, f'{year}_index')
    fnK = os.path.join(path1, 'K_vec.geo.tif')
    fnunw = os.path.join(path1, 'unwrapped_corr.geo.tif')
    fnref = os.path.join(path1, 'references_latlon.p')
    xy_ref = load_object(fnref)['regular']
    fntmp = os.path.join(path1, 'tmp_unwrapped_ref.p')

    if not os.path.exists(fntmp) or overwrite:
        K, geospatial_K = read_K(fnK)
        unw, geospatial_unw = read_geotiff_geospatial(fnunw)
        assert geospatial_unw == geospatial_K
        if remove_last:
            unw = unw[:-1, ...]
            K = K[:-1,:-1, ...]
        P = K.shape[0] + 1
        var_atmo = np.ones(P) * (caldict['var_rad'])  # in rad
        covmodel = RationalQuadraticSepDiagCovMV(caldict['l'], var_atmo, alpha=caldict['alpha'])

        folder_forcing = os.path.join(pathm1, os.pardir, os.pardir, 'forcing', 'kivalina')
        dailytemp, ind_scenes = kivalina_forcing(folder_forcing, year=year, remove_last=remove_last)

        fndist = os.path.join(path1, 'distance_cal.p')
        unw_cor, K_cor = spatial_referencing(
            unw, K, covmodel, xy_ref, geospatial_K, fndist=fndist, convert_to_length=False, overwrite=False)
        s_obs, K_s = length_conversion(unw_cor, K_cor, wavelength=wavelength, flip_sign=False)
        s = s_obs / np.cos(geom['ia'])  # vertical
        K_s = K_s / (np.cos(geom['ia']) ** 2)
        res = {
            's': s, 'K': K_s, 'ind_scenes': ind_scenes, 'd0': dailytemp.index[0],
            'geospatial': geospatial_unw}
        save_object(res, fntmp)
    else:
        res = load_object(fntmp)
    return res

def plot_profile_time_series(path0, scenario='2019r'):
    from scripts.plotting import (
        initialize_matplotlib, cmap_e, colslist, _get_index, ProfileInterpolator,
        contrast, add_arrow_line, plot_profile, add_scalebar, plot_profile_index)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import matplotlib.dates as mdates
    import matplotlib.lines as lines
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as transforms
    from datetime import timedelta, date
    from string import ascii_lowercase
    pathres = os.path.join(path0, scenario)
    # profile = ((-164.3881, 67.8120), (-164.4469, 67.8272))
    profile = ((-164.4236, 67.8357), (-164.3395, 67.7895))
    config = ('2019', 'TDD900_lastday')
    year = int(config[0])
    steps = 1024
    step_ts = (268, 940)  # (245, 980)

    fig = plt.figure()
    initialize_matplotlib()
    fig.set_size_inches((7.08, 2.50), forward=True)
    left, right = 0.12, 0.98
    base_annot = 0.95
    width, height, bottom = 0.36, 0.29, 0.08
    ruler = (0.40, 0.92)
    rects = [(left, 0.66, right - left, 0.24), (left, 0.53, right - left, 0.07),
             (left, bottom, width, height), (right - width, bottom, width, height)]
    axs = [fig.add_axes(rect) for rect in rects]

    ir = InversionResultsMmap.from_file(os.path.join(pathres, 'ir.p'))
    geospatial = ir.geospatial
    ygrid = ir.ygrid
    save_object(geospatial, os.path.join(pathres, 'geospatial.p'))
    save_object(ygrid, os.path.join(pathres, 'ygrid.p'))

    geospatial = load_object(os.path.join(pathres, 'geospatial.p'))
    ygrid = load_object(os.path.join(pathres, 'ygrid.p'))

    ft = load_object(os.path.join(path0, config[0], 'forcing_timing.p'))
    indranges = ft['indranges']
    indrange = indranges[ft['indranges_names'].index(config[1])]

    cmap = cmap_e
    elim = (0.0, 0.5)
    ymax = 0.70
    xticks = np.arange(7) * 1000

    # annotations
    annots = [(40, 'floodplain'), (260, 'colluvial--alluvial'), (470, 'rocky outcrop'),
              (750, 'colluvial--alluvial'), (980, 'floodplain')]

    
    # profile
    yf = load_object(os.path.join(pathres, 'yf_mean.npy'))
    e_mean = np.load(os.path.join(pathres, 'e_mean.npy'))
    plot_profile(
        axs[0], e_mean, geospatial, profile, ymax=ymax, vlim=elim, ygrid=ygrid, cmap=cmap,
        yf=yf[..., indrange], y_xlabel=None, steps=steps, yticks=(0.0, 0.2, 0.4, 0.6), x_ylabel=-0.04,
        xticks=xticks)
    axs[0].tick_params(labelbottom=False)
    axs[0].text(0.008, 0.02, 'a)', ha='left', va='baseline', transform=axs[0].transAxes, c='w')
    
    # index bar
    em = _read_config(config, indranges_names, None, path0)
    plot_profile_index(
        axs[1], em['mean'], geospatial, profile, cmap=cmap, vlim=elim, y_xlabel=-1.7, steps=steps,
        xticks=xticks)
    axs[1].tick_params(left=False, labelleft=False)
    axs[1].text(-0.015, 0.500, '$\\bar{e}$', ha='right', va='center', transform=axs[1].transAxes)
    axs[1].text(0.008, 0.350, 'b)', ha='left', va='center', transform=axs[1].transAxes, c='w')
    trans = transforms.blended_transform_factory(axs[1].transData, fig.transFigure)

    # annotations
    for step_annot, string_annot in annots:
        axs[0].text(step_annot, base_annot, string_annot, ha='center', va='baseline', transform=trans)

    # load unw and show two time series
    res_ts = _read_timeseries_kivalina(year=year, remove_last=config[1][-1] == 'r')
    pi = ProfileInterpolator(res_ts['geospatial'], profile[0], profile[1], steps=steps)
    s_profile = pi.interpolate(np.moveaxis(res_ts['s'], 0, -1))
    # axs[2].plot(np.arange(s_profile.shape[0]), s_profile[:, -1])
    K_profile = pi.interpolate(np.moveaxis(res_ts['K'], 0, -1))
    xy_profile = pi._xy
    _ind_to_date = lambda ind: (res_ts['d0'] + timedelta(days=int(ind))).to_numpy()
    dt_s = np.array([_ind_to_date(_ind) for _ind in res_ts['ind_scenes']])
    dt_indrange = [_ind_to_date(_ind) for _ind in indrange]
    # plot indrange

    s_lim, s_ticks = (-0.10, 0.01), (-0.10, -0.05, 0.00)
    dt_lim = (date(year, 5, 21), date(year, 9, 15))
    for jstep, step in enumerate(step_ts):
        ax = axs[2 + jstep]
        fig.add_artist(lines.Line2D((step,) * 2, ruler, transform=trans, zorder=0, lw=0.2, c='#666666'))
        fig.text(
            step + steps / 120, ruler[0], ascii_lowercase[jstep + 2] + ')', ha='left', va='baseline',
            transform=trans)
        xy_step = xy_profile[:, step]
        print(xy_step)
        stde = np.sqrt(np.diag(assemble_tril(K_profile[step,:])))
        s_step = np.concatenate(([0], s_profile[step,:]))
        rect = (dt_indrange[0], s_lim[0]), dt_indrange[1] - dt_indrange[0], s_lim[1] - s_lim[0]
        ax.add_patch(
            Rectangle(*rect, alpha=0.1, zorder=0, transform=ax.transData, color=colslist[2], ec='none'))

        ax.axhline(0.0, c='#cccccc', lw=0.5, alpha=0.5)
        ax.errorbar(dt_s[1:], s_step[1:], yerr=stde, linestyle='none', ecolor=colslist[0], elinewidth=0.5)
        ax.plot(dt_s, s_step, c=colslist[0])
        ax.set_ylim(s_lim)
        ax.set_yticks(s_ticks)
        ax.set_xlim(dt_lim)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.text(
            0.02, 0.02, ascii_lowercase[2 + jstep] + ')', ha='left', va='baseline', transform=ax.transAxes)

    axs[2].text(
        -0.19, 0.50, 'displacement [m]', ha='right', va='center', rotation=90, transform=axs[2].transAxes)

    # cbar
    cax0 = fig.add_axes([0.005, rects[1][1], 0.010, rects[0][1] + rects[0][3] - rects[1][1]])
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=Normalize(*elim, clip=True), cmap=cmap), cax=cax0, orientation='vertical')
    cbar.set_ticks([elim[0], elim[1] / 2, elim[1]])
    cax0.text(2.50, -0.26, '$e$ [$-$]', ha='center', va='baseline', transform=cax0.transAxes)

    # labels on top?

    plt.show()

if __name__ == '__main__':
    fnsubset = os.path.join(path0, 'subset.gpkg')
    fniceoptical = os.path.join(path0, 'iceoptical.gpkg')
    fncores = os.path.join(path0, 'cores2005.gpkg')
    ft = load_object(os.path.join(path0, '2019r', 'forcing_timing.p'))
    indranges_names = ft['indranges_names']
    configs = [
        ('2019', 'TDD900_lastday'), ('2019', 'TDD1000_lastday'), ('2018', 'TDD900_lastday'),
        ('2019r', 'TDD900_lastday')]
    config_labels = ['later scene', 'deeper', 'cooler summer', 'reference']

    # violin_plot(configs[-1], fncores=fncores)
    # fntmp = os.path.join(pathfig, 'kde.p')
    # plot_subset(configs, indranges_names, path0, config_labels=config_labels, fntmp=fntmp)
    plot_profile_time_series(path0, scenario='2019')

