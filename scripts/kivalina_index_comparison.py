'''
Created on Jun 28, 2023

@author: simon
'''

from pathlib import Path
import numpy as np
from rasterio.crs import CRS
import copy
from rasterio.transform import Affine

from analysis import (
    Geospatial, read_geotiff_geospatial, load_object, read_geotiff, K_from_K_vec, save_object,
    InversionResultsMmap, assemble_tril)

path0 = Path('/home/simon/Work/gie/processed/kivalina/index/')
pathfig = Path('/home/simon/Work/gie/figures/index/')
pathls, lsscene = Path('/home/simon/Work/gie/optical/Landsat/'), 'LC08_L2SP_083012_20190707_20200827_02_T1'
fnls = pathls / f'{lsscene}.vrt'
fnswir = pathls / lsscene, f'{lsscene}_SR_B7.TIF'
fndem = Path('/home/simon/Work/Kivalina/TDM90/Kivalina/DEM.tif')
fnforcing = Path('/home/simon/Work/Kivalina/forcing/T2MMEAN.csv')
fnpred = Path('/home/simon/Work/gie/ancillary/GEE/e_pred.tif')

profile = ((-164.4236, 67.8357), (-164.3395, 67.7895))

crs = CRS.from_epsg(3572)
posting = 50
transform = Affine(posting, 0.0, -615854, 0.0, -posting, -2381365)
shape = (124, 124)
geospatial_subset = Geospatial(transform, crs, shape=shape)

proj4_large = "+proj=laea +lat_0=90 +lon_0=-176.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
crs_large = CRS.from_proj4(proj4_large)
transform_large = Affine(posting, 0.0, 478800, 0.0, -posting, -2384000)
shape_large = (700, 1200)
geospatial_large = Geospatial(transform_large, crs_large, shape=shape_large)
transform_proc = Affine(posting, 0.0, 489274, 0.0, -posting, -2397437)
shape_proc = (310, 690)
geospatial_proc = Geospatial(transform_proc, crs_large, shape=shape_proc)

pathm1 = path0.parents[0]
_fnK = pathm1 / f'2019_index' / 'K_vec.geo.tif'
geospatial_native = Geospatial.from_file(path0 / _fnK)

from scripts.plotting import cmap_e
c_bad = '#444444'
cmap = copy.copy(cmap_e)
cmap.set_bad(color=c_bad)

def _normalize(im):
    anc = np.nanpercentile(im, (2, 99), axis=(1, 2))
    im -= anc[0,:, np.newaxis, np.newaxis]
    im /= (anc[1,:] - anc[0,:])[:, np.newaxis, np.newaxis]
    return np.moveaxis(im, 0, -1)

def resample_iceoptical(fniceoptical, geospatial):
    import geopandas as gpd
    iceoptical = gpd.read_file(fniceoptical).to_crs(geospatial.crs)
    iceoptical = iceoptical[iceoptical['include'] == 1]
    return geospatial.rasterize(iceoptical, field='code')


def resample_scenario(path0, scenario, geospatial, metrics=('mean', 'var'), apply_mask=True):
    geospatial_mean = load_object(path0 / scenario / 'ir.p')['geospatial']
    def _read(metric, mask=None):
        fnm = path0 / scenario / f'e_mean_period_{metric}.npy'
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

def mask_year(path0, year, thresh=1.25, opening=3, closing=20, geospatial_out=None):
    print(f'masking year {year}')
    pathm1 = path0.parents[0]
    fn = pathm1 / f'{year}_index' / 'K_vec.geo.tif'
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

def plot_subset(
        configs, indranges_names, path0, config_labels=None, fntmp=None, fnout=None, overwrite=False):
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
        if fntmp is None or not fntmp.exists() or overwrite:
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
    import matplotlib.lines as mlines
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as path_effects
    fig, axs = prepare_figure(
        nrows=3, ncols=len(configs), figsize=(1.8, 1.2), remove_spines=False, sharex='none', sharey='none',
        top=0.970, bottom=0.070, left=0.080, right=0.845, wspace=0.250, hspace=0.050)
    lims = {'mean': (0.0, 0.7), 'var': (0.00, 0.20)}
    e_ticks = {'mean': (0.0, 0.3, 0.6), 'var': (0.00, 0.10, 0.20)}
    e_lim = (0.00, 0.70)

    em, kd = _prepare_data()
    for jconfig, config in enumerate(configs):
        axs[0, jconfig].imshow(em[jconfig]['mean'], cmap=cmap, vmin=e_lim[0], vmax=e_lim[1])
        if jconfig < len(configs) - 1:
            for jmetric, metric in enumerate(metrics):
                ax, lim = axs[jmetric + 1, jconfig], lims[metric]
                ax.imshow(
                    kd[jconfig][metric].T, origin='lower', extent=lim * 2, cmap=cmap,
                    interpolation='nearest')
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
    cbarlabel = '$\\hat{\\bar{e}}$ [$-$]'
    cbar.solids.set_rasterized(True)
    cax0.text(1.50, vpos_label, cbarlabel, ha='center', va='baseline', transform=cax0.transAxes)

    # scale bar
    add_scalebar(axs[-1, -1], geospatial_subset, length=2e3, y=0.25, dx=-0.68, label='2 km', ylab=0.18)
    axs[-1, -1].text(
        1.03, 0.50, 'Landsat-8 true-color', rotation=270, ha='left', va='center',
        transform=axs[-1, -1].transAxes)
    y_simiq, x_simiq = 0.20, (-0.02, 0.02)
    axs[-1, -1].text(
        -0.03, y_simiq, 'Simiq', rotation=90, ha='right', va='center', transform=axs[-1, -1].transAxes)
    lineb = mlines.Line2D(
        x_simiq, (y_simiq,) * 2, transform=axs[-1, -1].transAxes, c='#cccccc', lw=0.8)
    linef = mlines.Line2D(
        x_simiq, (y_simiq,) * 2, transform=axs[-1, -1].transAxes, c='#666666', lw=0.5)

    fig.lines.extend([lineb, linef])

    ls, _ = geospatial_subset.warp_from_file(fnls)
    ls = ls[::-1,:,:]
    axs[2, -1].imshow(_normalize(ls))

    # labels
    if config_labels is not None:
        for jax, ax in enumerate(axs[0,:]):
            ax.text(0.50, 1.11, config_labels[jax], ha='center', va='baseline', transform=ax.transAxes)
    for ax in axs[-1,:-1]:
        ax.text(
            0.50, -0.33, 'baseline $\\mathrm{std}_{\\mathrm{p}}(\\hat{\\bar{e}})$ [$-$]', ha='center',
            va='baseline', transform=ax.transAxes)
    ylab = ('$\\hat{\\bar{e}}$ [$-$]', '$\\mathrm{std}_{\\mathrm{p}}(\\hat{\\bar{e}})$ [$-$]')
    for jax, ax in enumerate(axs[1:, 0]):
        ax.text(-0.33, 0.50, ylab[jax], rotation=90, ha='right', va='center', transform=ax.transAxes)
    # lines
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
    if fnout is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(fnout)

def rasterize_cores(fncores, geospatial):
    # breaks down when multiple coords map to the same pixel
    import geopandas as gpd
    cores = gpd.read_file(fncores).to_crs(geospatial.crs)
    imap_ = geospatial.rasterize(cores, field='code')
    mask_ = geospatial.rasterize(cores, field='include')
    imap_[mask_ == 0] = 255
    return imap_

def violin_plot(config, fncores=None, fnout=None):
    from scripts.plotting import colslist, prepare_figure
    imap = resample_iceoptical(fniceoptical, geospatial_subset)[0, ...]
    ft = load_object(path0 / config[0] / 'forcing_timing.p')
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
        0.5, 1.18, '$\\hat{\\bar{e}}$ [$-$]', transform=ax.transAxes,
        ha='center')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim((ypos[-1] - 0.4, ypos[0] + 0.5))
    if fnout is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(fnout)

def _read_timeseries_kivalina(year=2019, remove_last=False, overwrite=False):
    from scripts.kivalina_calibration import caldict
    from analysis import RationalQuadraticSepDiagCovMV, read_K, length_conversion, spatial_referencing
    from scripts.kivalina_index import kivalina_forcing, wavelength, geom

    path1 = pathm1 / f'{year}_index'
    fnK = path1 / 'K_vec.geo.tif'
    fnunw = path1 / 'unwrapped_corr.geo.tif'
    fnref = path1 / 'references_latlon.p'
    xy_ref = load_object(fnref)['regular']
    fntmp = path1 / 'tmp_unwrapped_ref.p'

    if not fntmp.exists() or overwrite:
        K, geospatial_K = read_K(fnK)
        unw, geospatial_unw = read_geotiff_geospatial(fnunw)
        assert geospatial_unw == geospatial_K
        if remove_last:
            unw = unw[:-1, ...]
            K = K[:-1,:-1, ...]
        P = K.shape[0] + 1
        var_atmo = np.ones(P) * (caldict['var_rad'])  # in rad
        covmodel = RationalQuadraticSepDiagCovMV(caldict['l'], var_atmo, alpha=caldict['alpha'])

        folder_forcing = pathm1.parents[1] / 'forcing' / 'kivalina'
        dailytemp, ind_scenes = kivalina_forcing(folder_forcing, year=year, remove_last=remove_last)

        fndist = path1 / 'distance_cal.p'
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

def plot_profile_time_series(path0, config, scenario='2019r', fnout=None):
    from scripts.plotting import (
        initialize_matplotlib, colslist, ProfileInterpolator, plot_profile, plot_profile_index)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import matplotlib.dates as mdates
    import matplotlib.lines as lines
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as transforms
    from datetime import timedelta, date
    from string import ascii_lowercase
    pathres = path0 / scenario
    year = int(config[0])
    steps = 1024
    step_ts = (272, 940)  # (268)(245, 980)

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

    # ir = InversionResultsMmap.from_file(pathres / 'ir.p')
    # geospatial = ir.geospatial
    # ygrid = ir.ygrid
    # save_object(geospatial, pathres / 'geospatial.p')
    # save_object(ygrid, pathres / 'ygrid.p')

    geospatial = load_object(pathres / 'geospatial.p')
    ygrid = load_object(pathres / 'ygrid.p')

    ft = load_object(path0 / config[0] / 'forcing_timing.p')
    indranges = ft['indranges']
    indrange = indranges[ft['indranges_names'].index(config[1])]

    elim = (0.0, 0.5)
    ymax = 0.70
    xticks = np.arange(7) * 1000

    # annotations
    annots = [(40, 'floodplain'), (260, 'colluvial--alluvial'), (470, 'rocky outcrop'),
              (750, 'colluvial--alluvial'), (980, 'floodplain')]

    # profile
    yf = load_object(pathres / 'yf_mean.npy')
    e_mean = np.load(pathres / 'e_mean.npy')
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
    axs[1].text(-0.015, 0.500, '$\\hat{\\bar{e}}$', ha='right', va='center', transform=axs[1].transAxes)
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
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def plot_regional(fnout=None):
    from scripts.plotting import add_scalebar, colslist, initialize_matplotlib
    import colorcet as cc
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from datetime import datetime as dt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from string import ascii_lowercase
    # fig, axs = prepare_figure(
    #     nrows=1, ncols=3, figsize=(2.0, 0.7), sharex=False, sharey=False, bottom=0.2, right=0.96, left=0.02,
    #     top=0.93, wspace=0.3, remove_spines=False)
    col_water = '#e6e6f1'
    initialize_matplotlib()
    fig = plt.figure()
    fig.set_size_inches((7.08, 2.20), forward=True)
    left, top = 0.01, 0.93
    width, height_s, height_f = 0.29, 0.55, 0.78
    hspace_l, hspace_r = 0.025, 0.065
    rects = [(left, top - height_s, width, height_s),
             (left + (width + hspace_l), top - height_s, width, height_s),
             (left + 2 * width + hspace_l + hspace_r, top - height_f, width, height_f)]

    geospatial = geospatial_large

    axs = [fig.add_axes(rect) for rect in rects]
    labels = ['Landsat true color', 'elevation', 'thawing degree days (TDD) [$^{\\circ}$C]']

    ax = axs[0]
    ls, _ = geospatial.warp_from_file(fnls)
    ls = ls[::-1,:,:]
    ax.imshow(_normalize(ls))
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    ccrsproj = ccrs.LambertAzimuthalEqualArea(
        central_longitude=-148, central_latitude=70)
    iax = fig.add_axes((left, 0.02, width, 0.34), projection=ccrsproj)
    crs_pc = ccrs.PlateCarree()
    iax.set_extent([-165, -135, 58, 71], crs=crs_pc)

    iax.add_feature(cfeature.OCEAN, color=col_water)
    iax.add_feature(cfeature.BORDERS, linestyle='-', lw=0.5, color='#cccccc')
    iax.plot(
        -164.54, 67.81, linestyle='none', marker='o', mfc=colslist[0], mec='none', ms=2,
        transform=crs_pc)
    iax.spines['geo'].set_linewidth(0.5)
    iax.spines['geo'].set_edgecolor('#666666')

    ax = axs[1]
    cmap_topo = copy.copy(cc.cm['CET_L10'])
    # dem, _ = geospatial_large.warp_from_file(fndem)
    thresh_swir = None
    dem, _ = geospatial.warp_from_file('/home/simon/Work/gie/ancillary/USGS_DEM/3DEP_DEM_Kivalina.tif')
    dem[dem <= 0.2] = -0.1  # np.nan
    cmap_topo.set_under(col_water)
    if thresh_swir is not None:
        from scipy.ndimage import binary_closing, binary_opening
        ls_swir, _ = geospatial.warp_from_file(fnswir)
        mask = (ls_swir < thresh_swir)[0, ...]
        mask = binary_opening((binary_closing(mask, iterations=1)), iterations=5)
        dem[mask[np.newaxis, ...]] = np.nan
    im = ax.imshow(dem[0, ...], cmap=cmap_topo, vmin=0.0, vmax=300, interpolation_stage='rgba')
    rc = geospatial.rowcol(
        np.array([geospatial_proc.transform.xoff, geospatial_proc.transform.yoff])[:, np.newaxis])
    rect = Rectangle(
        rc[:, 0], geospatial_proc.shape[1], geospatial_proc.shape[0], facecolor='none',
        edgecolor=colslist[0])
    ax.add_patch(rect)
    ax.text(rc[0, 0] + 50, rc[1, 0] - 30, 'study area', c=colslist[0])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    add_scalebar(ax, geospatial_proc, length=5000, label='5 km')
    cax = ax.inset_axes((0.00, -0.15, 0.50, 0.10))
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks((0, 100, 200, 300))
    cax.text(1.10, 0.20, '[m]', ha='left', va='center', transform=cax.transAxes)
    cbar.solids.set_rasterized(True)

    ax = axs[2]
    TDDs = (900,)
    TDDdict, cumdict = TDD_kivalina(fnforcing)
    for year in cumdict:
        T_y = cumdict[year][1]
        # print(year, T_y[-1] > 900, T_y[-1] > 1000, np.nonzero(T_y > 900)[0])
        from datetime import timedelta
        if year in (2018,):
            try:
                print(cumdict[year][0][0] + timedelta(days=int(np.nonzero(T_y > 900)[0][0])))
            except:
                raise
        if year in (2018, 2019):
            alpha, lw = 1.0, 0.8
            c = {2018: colslist[1], 2019: colslist[0]}[year]
            ax.text(len(T_y) + 3, T_y[-1], str(year), ha='left', va='center', c=c)
        else:
            alpha, lw, c = 0.4, 0.3, '#666666'
        ax.plot(np.arange(len(T_y)), T_y, alpha=alpha, lw=lw, c=c)
    period = cumdict[year][0]
    months = [5, 6, 7, 8, 9]
    ax.set_xticks([(dt(year, m, 1) - period[0]).days for m in months])
    ax.set_xticklabels([f'{m:02}-01' for m in months])
    ax.set_yticks((0, 300, 600, 900, 1200))
    for TDD in TDDs:
        ax.axhline(TDD, lw=0.2, c='#666666')
        ax.text(1, TDD + 25, f'{TDD}' + '\\,$^{\\circ}$C\\,d threshold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.text(0.50, -0.18, 'date [mm-dd]', ha='center', va='baseline', transform=ax.transAxes)

    import matplotlib.transforms as transforms
    for jlabel, label in enumerate(labels):
        ax = axs[jlabel]
        lab = f"{ascii_lowercase[jlabel]}) {label}"
        trans_b = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
        ax.text(0.00, 0.95, lab, ha='left', va='baseline', transform=trans_b)

    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def read_timeseries(fn, first_line=9, dtformat='%Y-%m-%d', offset=0.0):
    from datetime import datetime
    with open (fn, 'r') as f:
        raw = f.readlines()[first_line:]
    dt, val = [], []
    for l in raw:
        try:
            val_ = l.strip().split(',')
            dt.append(datetime.strptime(val_[0], dtformat))
            val.append(float(val_[1]) + offset)
        except:
            pass
    return dt, np.array(val)

def _TDD(dt, TC, days_snow=14):
    from datetime import datetime
    period = lambda year: (datetime(year, 4, 25), datetime(year, 9, 21))
    def _filter_T(year):
        _period = period(year)
        ind = np.array([(dt_.year == year and T_ > 0.0 and dt_ >= _period[0] and dt_ <= _period[1])
                        for dt_, T_ in zip(dt, TC)])
        if days_snow is not None:
            melt_onset = np.nonzero(ind)[0][0]
            ind[melt_onset:melt_onset + days_snow] = False
        return ind
    TDDdict = {}
    cumTDDdict = {}
    years = set([dt_.year for dt_ in dt])
    for y in years:
        _period, ind_y = period(y), _filter_T(y)
        TDDdict[y] = sum(TC[ind_y])
        _T, _dt = TC.copy(), np.array(dt)
        _T[np.logical_not(ind_y)] = 0
        T_period = _T[np.logical_and(_dt >= _period[0], _dt <= _period[1])]
        cumTDDdict[y] = (_period, np.cumsum(T_period))
    return TDDdict, cumTDDdict

def TDD_kivalina(fnforcing):
    dt, TC = read_timeseries(fnforcing, offset=-273.15)
    TDDdict, cumTDDdict = _TDD(dt, TC)
    return TDDdict, cumTDDdict

def rc_references(path0, config, geospatial):
    pathm1 = path0.parents[0]
    path1 = pathm1 / f'{config[0]}_index'
    fnref = path1 / 'references_latlon.p'
    xy_ref = load_object(fnref)['regular']
    rc_ref = geospatial.rowcol(xy_ref, crs='EPSG:4326')
    return rc_ref

def plot_atmosphere(config_ref, config_r, path0, indranges_names, fnout=None):
    from scripts.plotting import prepare_figure, add_scalebar
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from string import ascii_lowercase

    e_lim = (0.00, 0.70)
    std_lim = (0.00, 0.20)
    rc_ref = rc_references(path0, config_ref, geospatial_proc)
    fig, axs = prepare_figure(
        nrows=2, ncols=2, figsize=(1.00, 0.47), top=0.920, bottom=0.020, right=0.88, left=0.010,
        hspace=0.10, wspace=0.06, remove_spines=False)

    e_ref = _read_config(config_ref, indranges_names, geospatial_proc, path0)
    e_r = _read_config(config_r, indranges_names, geospatial_proc, path0)
    for jres, res in enumerate((e_ref, e_r)):
        axs[0, jres].imshow(
            res['mean'], cmap=cmap, vmin=e_lim[0], vmax=e_lim[1], interpolation='nearest')
        axs[1, jres].imshow(
            np.sqrt(res['var']), cmap=cmap, vmin=std_lim[0], vmax=std_lim[1], interpolation='nearest')

    ylab = 0.16
    for jax, ax in enumerate(axs.flatten()):
        ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        ax.text(
            0.02, ylab, f'{ascii_lowercase[jax]})', ha='left', va='top', c='#dddddd', transform=ax.transAxes)
    add_scalebar(
        axs[0, 0], geospatial_proc, length=5e3, label='5 km', color='#dddddd', y=0.21, dx=0.72, ylab=ylab)
    cax_extent = [1.04, 0.06, 0.06, 0.60]
    cbarlabels = ('$\\hat{\\bar{e}}$ [$-$]', '$\\mathrm{std}_{\\mathrm{p}}\\hat{\\bar{e}}$')
    c, lw, ec, s = 'none', 0.5, 'w', 3
    axs[0, 0].scatter(rc_ref[1,:], rc_ref[0,:], c=c, s=s, linewidths=lw, edgecolors=ec)
    axs[0, 1].scatter(rc_ref[1, 5], rc_ref[0, 5], c=c, s=s, linewidths=lw, edgecolors=ec)
    for jrow, lim in enumerate((e_lim, std_lim)):
        cax = axs[jrow, -1].inset_axes(cax_extent)
        cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(*lim, clip=True), cmap=cmap), cax=cax)
        cbar.set_ticks([lim[0], lim[1]])
        cbar.solids.set_rasterized(True)
        cax.text(0.00, 1.31, cbarlabels[jrow], ha='left', va='baseline', transform=cax.transAxes)
    collabels = ('baseline', 'single reference')
    for jcol, collabel in enumerate(collabels):
        axs[0, jcol].text(
            0.50, 1.05, collabel, ha='center', va='baseline', transform=axs[0, jcol].transAxes)
    if fnout is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(fnout, dpi=450)

def plot_index(config, path0, fnls, fnout=None, _cmap=None):
    from scripts.plotting import prepare_figure, add_scalebar, colslist
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as path_effects
    from string import ascii_lowercase
    if _cmap is None: _cmap = cmap

    e_lim = (0.00, 0.70)
    rc_ref = rc_references(path0, config, geospatial_proc)
    fig, axs = prepare_figure(
        nrows=1, ncols=2, figsize=(2.05, 0.55), top=0.990, bottom=0.160, right=0.985, left=0.015,
        hspace=0.10, wspace=0.06, remove_spines=False)

    e_res = _read_config(config, indranges_names, geospatial_proc, path0)
    axs[0].imshow(e_res['mean'], cmap=_cmap, vmin=e_lim[0], vmax=e_lim[1], interpolation='nearest')

    ls, _ = geospatial_proc.warp_from_file(fnls)
    ls = ls[::-1,:,:]
    axs[1].imshow(_normalize(ls))
    # show cores
    import geopandas as gpd
    gdf = gpd.read_file(fncores).to_crs(geospatial_proc.crs)
    gdf = gdf[gdf['include'] == 1]
    rc_cores = geospatial_proc.rowcol(gdf)
    s_core, c_core = 2, '#ffdf1d'
    axs[1].scatter(
        rc_cores[1,:], rc_cores[0,:], c=c_core, s=2, edgecolors='none')
    # show focus region
    gss = geospatial_subset.shape
    rc_subset = np.array([[0, 0], [gss[0], 0], [gss[0], gss[1]], [0, gss[1]]]).T
    xy_subset = geospatial_subset.xy(rc_subset)
    rcp = geospatial_proc.rowcol(xy_subset, crs=geospatial_subset.crs)
    c_subset = colslist[2]
    for js in range(rcp.shape[1]):
        je = (js + 1) % (rcp.shape[1])
        axs[1].plot(
            (rcp[1, js], rcp[1, je]), (rcp[0, js], rcp[0, je]), c=c_subset, lw=0.8)
    txt = axs[1].text(0.86, 0.43, 'subset', c=c_subset, transform=axs[1].transAxes)
    txt.set_path_effects(
        [path_effects.Stroke(linewidth=1.5, foreground='#333333'), path_effects.Normal()])

    # show profile
    from scripts.plotting import add_arrow_line
    rc_profile = geospatial_proc.rowcol(np.array(profile).T, crs='EPSG:4326')
    add_arrow_line(
        axs[1], rc_profile, c='#f35092', hwidth=12, hlength=18, pos_frac=[0.75, 0.30], label='T',
        dlabel=(-10, 30))

    for jax, ax in enumerate(axs.flatten()):
        ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        label = f'{ascii_lowercase[jax]})'
        txt = ax.text(0.01, 0.02, label, ha='left', va='baseline', c='#dddddd', transform=ax.transAxes)

    c, lw, ec, s = 'none', 0.5, 'w', 3
    axs[0].scatter(rc_ref[1,:], rc_ref[0,:], c=c, s=s, linewidths=lw, edgecolors=ec)
    cax_left, cax_height, cax_top = 0.04, 0.06, -0.04
    cax = axs[0].inset_axes((cax_left, cax_top - cax_height, 0.40, cax_height))
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=Normalize(*e_lim, clip=True), cmap=_cmap), cax=cax, orientation='horizontal')
    cbar.set_ticks([e_lim[0], e_lim[1] / 2, e_lim[1]])
    cbar.solids.set_rasterized(True)
    cax.text(1.07, 0.30, '$\\hat{\\bar{e}}$ [$-$]', ha='left', va='center', transform=cax.transAxes)
    add_scalebar(
        axs[0], geospatial_proc, length=5e3, label='5 km', y=cax_top, dx=cax_left,
        ylab=cax_top - cax_height)

    # legend: cores
    lax = axs[1].inset_axes((0, cax_top - cax_height, 1.00, cax_height))
    lax.scatter(
        cax_left, 0.50, s=s_core * 2, c=c_core, transform=lax.transAxes, linewidths=0.3,
        edgecolors=colslist[0])
    ax.text(0.08, 0.40, 'cores', ha='left', va='center', transform=lax.transAxes)
    lax.axis('off')
    if fnout is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(fnout, dpi=450)

def plot_rf_map(config_ref, fnpred, fnls, path0, indranges_names, fnout=None):
    from scripts.plotting import prepare_figure, add_scalebar
    from matplotlib import cm
    from matplotlib.colors import Normalize

    e_lim = (0.00, 0.70)
    fig, axs = prepare_figure(
        nrows=1, ncols=3, figsize=(2.03, 0.32), top=0.930, bottom=0.000, right=0.925, left=0.0100,
        hspace=0.10, wspace=0.12, remove_spines=False)

    e_ref = _read_config(config_ref, indranges_names, geospatial_proc, path0)['mean']
    e_pred, _ = geospatial_proc.warp_from_file(fnpred)
    e_pred = e_pred[0, ...]
    e_pred[np.isnan(e_ref)] = np.nan
    for jres, res in enumerate((e_pred, e_ref)):
        axs[jres].imshow(
            res, cmap=cmap, vmin=e_lim[0], vmax=e_lim[1], interpolation='nearest')

    ls, _ = geospatial_proc.warp_from_file(fnls)
    ls = ls[::-1,:,:]
    axs[2].imshow(_normalize(ls))

    ylab = 0.13
    for ax in axs.flatten():
        ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)

    add_scalebar(
        axs[0], geospatial_proc, length=5e3, label='5 km', color='#dddddd', y=0.19, dx=0.80, ylab=ylab)

    cax_extent = [1.053, 0.120, 0.033, 0.600]
    cbarlabel = '$\\bar{e}$ [$-$]'
    cax = axs[-1].inset_axes(cax_extent)
    cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(*e_lim, clip=True), cmap=cmap), cax=cax)
    cbar.set_ticks((0.0, 0.3, 0.6))
    cbar.solids.set_rasterized(True)
    cax.text(0.00, 1.20, cbarlabel, ha='left', va='baseline', transform=cax.transAxes)

    collabels = ('f) random forest $\\bar{e}$', 'g) InSAR $\\hat{\\bar{e}}$', 'h) Landsat-8 true-color ')
    for jcol, collabel in enumerate(collabels):
        axs[jcol].text(
            0.005, 1.050, collabel, ha='left', va='baseline', transform=axs[jcol].transAxes)
    if fnout is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(fnout, dpi=450)

def plot_index_cbars(config, path0, fnls):
    import colorcet as cc
    from matplotlib.colors import LinearSegmentedColormap
    cmapnames = ['CET_CBC1']
    # cmapnames = ['bgy', 'bmw', 'bmy', 'CET_CBL1', 'CET_L4', 'kgy', 'CET_L16', 'CET_CBL2', 'CET_CBC1']
    for cmapname in cmapnames:
        if cmapname != 'CET_CBC1':
            _cmap = copy.copy(cc.cm[cmapname])
            _cmap.set_bad(color=c_bad)
        else:
            cmap = cc.cm[cmapname]
            _cmap = LinearSegmentedColormap.from_list('clipped', cmap(np.linspace(0.8, 0.2, 256)))
            _cmap.set_bad(color='#666666')
        plot_index(config, path0, fnls, fnout=pathfig / f'index_{cmapname}.pdf', _cmap=_cmap)

if __name__ == '__main__':
    fnsubset = path0 / 'subset.gpkg'
    fniceoptical = path0 / 'iceoptical.gpkg'
    fncores = path0 / 'cores2005.gpkg'
    ft = load_object(path0 / '2019r' / 'forcing_timing.p')
    indranges_names = ft['indranges_names']
    configs = [
        ('2019', 'TDD900_lastday'), ('2019', 'TDD1000_lastday'), ('2018', 'TDD900_lastday'),
        ('2019r', 'TDD900_lastday')]
    config_labels = ['extra scene', 'later $\\bar{e}$', '2018', 'baseline']

    # violin_plot(configs[-1], fncores=fncores, fnout=pathfig / 'violin.pdf')
    # fntmp = pathfig / 'kde.p'
    # plot_subset(
    #     configs, indranges_names, path0, config_labels=config_labels, fntmp=fntmp,
    #     fnout=pathfig / 'subset.pdf', overwrite=False)
    plot_profile_time_series(path0, configs[0], scenario='2019', fnout=pathfig / 'profile.pdf')
    # plot_regional(fnout=pathfig / 'regional.pdf')

    # plot_atmosphere(
    #     configs[0], ('2019rs', 'TDD900_lastday'), path0, indranges_names,
    #     fnout=pathfig / 'atmos.pdf')
    # plot_index(configs[0], path0, fnls, fnout=pathfig / 'index.pdf')
    # plot_rf_map(configs[0], fnpred, fnls, path0, indranges_names, fnout=pathfig / 'RFmap.pdf')
    # plot_index_cbars(configs[0], path0, fnls)
