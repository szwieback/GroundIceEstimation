'''
Created on Aug 29, 2025

@author: simon
'''
import copy
import numpy as np
import matplotlib.pyplot as plt
from analysis import load_object
from scripts.pathnames import paths
from scripts.plotting import cmap_e, prepare_figure, colslist, add_scalebar
import colorcet as cc
from matplotlib.colors import ListedColormap
from collections import namedtuple

c_bad = '#555555'
cmap = copy.copy(cmap_e)
cmap.set_bad(color=c_bad)

MetricSummary = namedtuple('MetricSummary', ['accuracy', 'sharpness', 'coverage'])
methodlabels = {'GM_K1': 'GM: $K=1$', 'GM_K2': 'GM: $K=2$', 'GM_K3': 'GM: $K=3$', 'GM_K4': 'GM: $K=4$',
                'GM_K5': 'GM: $K=5$', 'IS': 'IS'}

jdepth = 2  # which one to show

lonlats = {'HV': (-148.8437, 69.1548), 'IC': (-148.8304, 69.0403), 'HVE': (-148.8373, 69.1559)}


def load_results(p0, imethod, ftype, rmethod='mintpy'):
    arr = np.load(p0 / f'{rmethod}_{imethod}' / f'{ftype}.npy')
    return arr

def load_geospatial(p0, imethod, rmethod='mintpy'):
    from analysis import MulticlassInversionResultsISMmap as IR
    from analysis import save_object, load_object
    fnir = p0 / f'{rmethod}_{imethod}' / 'ir.p'
    fngeospatial = p0 / f'{rmethod}_{imethod}' / 'geospatial.p'
    if not fngeospatial.exists():
        ir = IR.from_file(fnir)
        geospatial = ir.geospatial
        save_object(geospatial, fngeospatial)
    else:
        geospatial = load_object(fngeospatial)
    return geospatial

def plot_results(p0, imethods, fnout=None):
    from string import ascii_lowercase
    import matplotlib.patheffects as path_effects
    
    fig, axs = prepare_figure(
        nrows=2, ncols=5, figsize=(2.00, 1.28), hspace=0.09, bottom=0.02, top=0.96, left=0.07,
        remove_spines=False, sharex=True, sharey=True)
    for jim, imethod in enumerate(imethods):
        e_mean = load_results(p0, imethod, 'e_mean_period_mean')
        im = axs[0, jim].imshow(e_mean[..., 0], vmin=0.0, vmax=0.40, cmap=cmap, interpolation='nearest')
        e_mean = load_results(p0, imethod, 'e_mean_depth_mean')
        axs[1, jim].imshow(e_mean[..., jdepth], vmin=0.0, vmax=0.40, cmap=cmap, interpolation='nearest')
        axs[0, jim].text(
            0.50, 1.02, methodlabels[imethod], c='k', transform=axs[0, jim].transAxes, ha='center',
            va='baseline')
    S2, geospatial = load_S2(p0)
    rgb = np.stack([percentile_stretch(S2[b]) for b in [4, 3, 2]], axis=-1)
    axs[1, -1].imshow(rgb)
    site = 'HV'
    rc = geospatial.rowcol(lonlats[site], crs='epsg:4326')
    axs[1, -1].plot(
        rc[1], rc[0], linestyle='none', marker='o', mec='w', mfc=colslist[0], mew=0.8, ms=3.0)
    text = axs[1, -1].text(rc[1] - 5, rc[0] + 1, site, ha='right', va='center', c=colslist[0])
    text.set_path_effects([path_effects.withStroke(linewidth=0.8, foreground='w')])
    add_scalebar(axs[1, -1], geospatial, length=2000, label='2 km', y=0.97, dx=0.75, ylab=0.94)

    axs[0, 0].set_xticks((80,))
    axs[0, 0].set_yticks(np.array(range(4)) * 80 + 40)
    # legend
    ax = axs[0, 4]
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=1.0, aspect=5)
    cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])  # Set specific tick locations
    cbar.ax.text(
        2.05, 0.50, '$\\bar{e}$ or $e$ [$-$]', rotation=270, transform=cbar.ax.transAxes, ha='right',
        va='center')
    cbar.ax.text(
        2.65, 0.50, 'excess ice parameter', rotation=270, transform=cbar.ax.transAxes, ha='right',
        va='center')
    for ax, label in zip(axs[:, 0], ('time average $\\bar{e}$', 'depth average $e_{20-30}$')):
        ax.text(-0.14, 0.50, label, rotation=90, transform=ax.transAxes, c='k', va='center', ha='right')
    for jax, ax in enumerate(axs.flat):
        _c = '#dddddd' if jax != 9 else '#666666'
        ax.grid(True, alpha=0.6, color=_c, linewidth=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if jax != 4:
            _jax = jax if jax < 4 else jax - 1
            label = f'{ascii_lowercase[_jax]})'
            ax.text(0.98, 0.02, label, ha='right', va='baseline', transform=ax.transAxes, c=_c)
    c_ann = '#cccccc'
    ax = axs[0, 1]
    ax.text(0.02, 0.27, 'D', c=c_ann, transform=ax.transAxes)
    ax.text(0.65, 0.47, 'D', c=c_ann, transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout, dpi=450)

def plot_results_quantile(p0, imethods, fnout=None):
    from string import ascii_lowercase
    cmap_disp = copy.copy(cc.cm['CET_L18'])
    cmap_disp.set_bad(color='#aaaaaa')
    fig, axs = prepare_figure(
        nrows=1, ncols=4, figsize=(1.00, 0.42), hspace=0.09, bottom=0.00, top=0.93, left=0.01, right=0.85,
        remove_spines=False, sharex=True, sharey=True)
    for jim, imethod in enumerate(imethods):
        e_q = load_results(p0, imethod, 'e_mean_period_quantile')
        im = axs[jim].imshow(
            e_q[..., 0, 1] - e_q[..., 0, 0], vmin=0.0, vmax=0.48, cmap=cmap_disp, interpolation='nearest')
        axs[jim].text(
            0.50, 1.02, methodlabels[imethod], c='k', transform=axs[jim].transAxes, ha='center',
            va='baseline')
    # legend
    cax = fig.add_axes([0.86, 0.05, 0.02, 0.84])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(pad=2)
    cbar.set_ticks([0.0, 0.2, 0.4])    
    cbar.ax.text(
        6.80, 0.50, '$\\bar{e}$ CI width [$-$]', rotation=270, transform=cbar.ax.transAxes, ha='right',
        va='center')
    axs[0, 0].set_xticks((80,))
    axs[0, 0].set_yticks(np.array(range(4)) * 80 + 40)    
    for jax, ax in enumerate(axs.flat):
        _c = '#ffffff' if jax != 9 else '#666666'
        ax.grid(True, alpha=0.6, color=_c, linewidth=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        label = f'{ascii_lowercase[jax]})'
        ax.text(0.98, 0.92, label, ha='right', va='baseline', transform=ax.transAxes, c=_c)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout, dpi=450)

def percentile_stretch(band, low_percentile=2, high_percentile=98):
    p_low, p_high = np.nanpercentile(band, [low_percentile, high_percentile])
    stretched = np.clip(band, p_low, p_high)
    return (stretched - p_low) / (p_high - p_low)

def load_S2(p0):
    ir = load_object(p0 / f'mintpy_IS' / 'ir.p')
    geospatial = ir['geospatial']
    fnS2 = paths['ancillary'] / 'Sentinel2' / 'S2_composite_Dalton.tif'
    S2 = geospatial.warp_from_file(fnS2)
    return S2

def plot_comparison(p0, imethods, imethod_ref, fnout=None):
    from scipy.stats import gaussian_kde
    from string import ascii_lowercase
    fig, axs = prepare_figure(nrows=2, ncols=len(imethods), figsize=(1.00, 0.64), wspace=0.28, bottom=0.17,
                              hspace=0.25, left=0.13, right=0.98, top=0.92)
    e_mean_period_ref = load_results(p0, imethod_ref, 'e_mean_period_mean')
    e_mean_depth_ref = load_results(p0, imethod_ref, 'e_mean_depth_mean')
    e_mean_period_ref_var = load_results(p0, imethod_ref, 'e_mean_period_var')
    e_mean_depth_ref_var = load_results(p0, imethod_ref, 'e_mean_depth_var')
    lim = (-0.03, 0.40)
    gray_cmap = cc.cm.gray
    colors = gray_cmap(np.linspace(0.7, 0.0, 256))
    custom_cmap = ListedColormap(colors)
    def add_panel(ax, x, y, variance=None):
        stride = 37  # prime number to avoid subsampling depths
        _xy = np.vstack([x.flatten(), y.flatten()])
        xy = _xy[:, np.all(np.isfinite(_xy), axis=0)][:,::stride]
        if variance is not None:
            var = variance.flatten()[np.all(np.isfinite(_xy), axis=0)][::stride]
            score = np.median(np.abs((xy[1,:] - xy[0,:])) / np.sqrt(var))
        kde = gaussian_kde(xy)
        density = kde(xy)
        ax.plot(lim, lim, c='#cccccc', lw=0.3, zorder=3)
        ax.scatter(xy[0,:], xy[1,:], c=density, cmap=custom_cmap, alpha=0.7, s=1, zorder=4, linewidths=0,
                   vmin=0, vmax=np.percentile(density, 80), rasterized=True)
        if variance is not None:
            ax.text(
                0.03, 0.88, f'$\\Delta = {score:.1f}$', ha='left', va='baseline', transform=ax.transAxes)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    for jim, imethod in enumerate(imethods):
        e_mean_period = load_results(p0, imethod, 'e_mean_period_mean')
        add_panel(axs[0, jim], e_mean_period_ref, e_mean_period, variance=e_mean_period_ref_var)
        e_mean_depth = load_results(p0, imethod, 'e_mean_depth_mean')
        add_panel(axs[1, jim], e_mean_depth_ref, e_mean_depth, variance=e_mean_depth_ref_var)
    for ax, lab in zip(axs[:, 0], ['GM $\\bar{e}$ [$-$]', 'GM $e$ [$-$]']):
        ax.text(
            -0.49, 0.50, lab, ha='right', va='center', transform=ax.transAxes, rotation=90)
    for ax in axs[-1,:]:
        ax.text(0.50, -0.34, 'IS $\\bar{e}$ or $e$ [$-$]', ha='center', va='top', transform=ax.transAxes)
    for ax, imethod in zip(axs[0,:], imethods):
        ax.text(
            0.50, 1.09, methodlabels[imethod], ha='center', va='baseline', transform=ax.transAxes, c='k')
    for jax, ax in enumerate(axs.flatten()):
        ax.text(1.00, 0.05, f'{ascii_lowercase[jax]})', ha='right', va='baseline', transform=ax.transAxes)
    if fnout is not None:
        fig.savefig(fnout, dpi=450)
    else:
        plt.show()

def plot_comparison_quantile(p0, imethods, imethod_ref, fnout=None):
    from scipy.stats import gaussian_kde
    from string import ascii_lowercase
    fig, axs = prepare_figure(nrows=2, ncols=len(imethods), figsize=(1.00, 0.64), wspace=0.28, bottom=0.17,
                              hspace=0.25, left=0.13, right=0.98, top=0.92)
    e_mean_period_ref = load_results(p0, imethod_ref, 'e_mean_period_quantile')
    e_mean_depth_ref = load_results(p0, imethod_ref, 'e_mean_depth_quantile')
    lim = (-0.03, 0.40)
    gray_cmap = cc.cm.gray
    colors = gray_cmap(np.linspace(0.7, 0.0, 256))
    custom_cmap = ListedColormap(colors)
    def add_panel(ax, x, y):
        _x = x[..., 1] - x[..., 0]
        _y = y[..., 1] - y[..., 0]
        xy = np.vstack([_x.flatten(), _y.flatten()])
        xy = xy[:, np.all(np.isfinite(xy), axis=0)][:,::11]  # 11  # prime number to avoid subsampling depths
        kde = gaussian_kde(xy)
        density = kde(xy)
        ax.plot(lim, lim, c='#cccccc', lw=0.3, zorder=3)
        ax.scatter(xy[0,:], xy[1,:], c=density, cmap=custom_cmap, alpha=0.7, s=1, zorder=4, linewidths=0,
                   vmin=0, vmax=np.percentile(density, 80), rasterized=True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    for jim, imethod in enumerate(imethods):
        e_mean_period = load_results(p0, imethod, 'e_mean_period_quantile')
        add_panel(axs[0, jim], e_mean_period_ref, e_mean_period)
        e_mean_depth = load_results(p0, imethod, 'e_mean_depth_quantile')
        add_panel(axs[1, jim], e_mean_depth_ref, e_mean_depth)
    for ax, lab in zip(axs[:, 0], ['GM $\\bar{e}$ [$-$]', 'GM $e$ [$-$]']):
        ax.text(
            -0.49, 0.50, lab, ha='right', va='center', transform=ax.transAxes, rotation=90)
    for ax in axs[-1,:]:
        ax.text(0.50, -0.34, 'IS $\\bar{e}$ or $e$ [$-$]', ha='center', va='top', transform=ax.transAxes)
    for ax, imethod in zip(axs[0,:], imethods):
        ax.text(
            0.50, 1.09, methodlabels[imethod], ha='center', va='baseline', transform=ax.transAxes, c='k')
    for jax, ax in enumerate(axs.flatten()):
        ax.text(1.00, 0.05, f'{ascii_lowercase[jax]})', ha='right', va='baseline', transform=ax.transAxes)
    if fnout is not None:
        fig.savefig(fnout, dpi=450)
    else:
        plt.show()

def summary_metrics(variable, imethod):
    metrics = load_object(paths['simulation'] / f'sagwon_comparison_{imethod}' / f'metrics_{variable}.p')
    rmse = np.nanmean(metrics['RMSE'], axis=0)
    sharpness = np.sqrt(np.nanmean(metrics['variance'], axis=0))
    coverage = np.nanmean(metrics['coverage'][..., 1], axis=0)
    return MetricSummary(rmse, sharpness, coverage)

def plot_metrics(imethods, fnout=None):
    from string import ascii_lowercase
    fig, axs = prepare_figure(ncols=3, nrows=1, figsize=(1.00, 0.55), sharey=True, top=0.89, bottom=0.19)
    M = 6
    y = np.array(np.arange(M)).astype(np.float32)
    y[-1] = y[-1] + 0.4
    uoffset = 0.08
    offsets = {
        'GM_K1': 0 * uoffset, 'GM_K2':0 * uoffset, 'GM_K3':-1 * uoffset , 'GM_K5': 1 * uoffset, 'IS': 0}
    markers = {
        'GM_K1': 'o', 'GM_K2': 'o', 'GM_K3': 'o', 'GM_K5':'o', 'IS': '|'}
    markersizes = {'GM_K1': 4, 'GM_K2': 4, 'GM_K3': 4, 'GM_K5': 4, 'IS': 8}
    mews = {'GM_K1': 0.9, 'GM_K2': 0.9, 'GM_K3': 0.9, 'GM_K5': 0.9, 'IS': 1.4}
    colors = {
        'GM_K1': '#cc9999', 'GM_K2': colslist[2], 'GM_K3': colslist[0], 'GM_K5': colslist[1], 'IS': '#999999'}
    legend_artists = []
    def _plot(ax, mdepth, mind, imethod):
        m = np.concatenate((mdepth, mind))
        _y = y + offsets[imethod]
        l = ax.plot(m, _y, linestyle='none', marker=markers[imethod], markersize=markersizes[imethod],
                    mew=mews[imethod], mfc='none', c=colors[imethod])
        return l[0]
    for imethod in imethods:
        msd = summary_metrics('e_depthranges', imethod)
        isd = summary_metrics('e_indranges', imethod)
        l = _plot(axs[0], msd.accuracy, isd.accuracy, imethod)
        legend_artists.append(l)
        axs[0].set_xlim(0.00, 0.18)
        _plot(axs[1], msd.sharpness, isd.sharpness, imethod)
        axs[1].set_xlim(0.00, 0.18)

        axs[2].axvline(0.80, c='#cccccc', lw=0.5, alpha=0.5)
        _plot(axs[2], msd.coverage, isd.coverage, imethod)
        axs[2].set_xlim(0.65, 0.95)
    axs[0].set_ylim(M + 0.0, -0.25)
    xlabels = ['RMSE [$-$]', '$\\sigma_{\\mathrm{p}}$ [$-$]', 'coverage [$-$]']
    axs[0].set_yticks(y)
    yticklabels = ['$e_{0-10}$', '$e_{10-20}$', '$e_{20-30}$',
                    '$e_{30-40}$', '$e_{40-50}$', '$\\bar{e}$']
    axs[0].set_yticklabels(yticklabels)
    fig.legend(legend_artists, [methodlabels[imethod] for imethod in imethods], loc='upper center', ncol=4,
               handletextpad=0.03, columnspacing=0.52, bbox_to_anchor=(0.50, 1.03), frameon=False)
    for ax, xlabel in zip(axs, xlabels):
        ax.text(0.50, -0.24, xlabel, ha='center', va='baseline', transform=ax.transAxes)
    for jax, ax in enumerate(axs):
        ax.text(1.00, 0.02, f'{ascii_lowercase[jax]})', ha='right', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def _plot_core(ax, y_grid, e_mean, e_quantile=None, c=None, alpha=1.0, alpha_quantile=0.2, lw=1.1):
    if c is None: c = 'k'
    delta_y = y_grid[1] - y_grid[0]
    if e_quantile is not None:
        ax.fill_betweenx(
            y_grid, e_quantile[0,:], e_quantile[1,:], step='post', color=c, alpha=alpha_quantile, ec='none')
    ax.step(e_mean, y_grid, alpha=alpha, c=c, lw=lw)
    ax.plot((e_mean[-1],) * 2, (y_grid[-1], y_grid[-1] + delta_y), alpha=alpha, lw=lw, c=c)

def _add_legend_inset(
        legend_ax, c_is, c_im, alpha, alpha_q, alpha_shade, lw, lw_q, 
        bbox_to_anchor=(0.70, 0.75, 0.40, 0.30)):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    inset = inset_axes(legend_ax, width=0.25, height=0.30,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=legend_ax.transAxes, loc='upper right')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines:
        inset.spines[spine].set_visible(False)
    x = [0, 1]
    x_line = [0.00, 0.95]
    y_is = (0.3, )  * 2
    y_offset = 0.1
    x_text = 1.4
    inset.fill_between(
        x, (y_is[0] - y_offset, )*2, (y_is[0] + y_offset, )*2, color=c_is, alpha=alpha_shade, ec='none')
    inset.plot(x_line, y_is, color=c_is, linewidth=lw, alpha=alpha)
    inset.text(x_text, y_is[0], 'in-situ', va='center', fontsize=7)
    y_mean = (0.7,)*2
    y_q1 = (y_mean[0]-y_offset,)*2  
    y_q2 = (y_mean[0]+y_offset,)*2 
    inset.plot(x_line, y_mean, color=c_im, linewidth=lw, alpha=alpha)
    inset.plot(x, y_q1, color=c_im, linewidth=lw_q, alpha=alpha_q)
    inset.plot(x, y_q2, color=c_im, linewidth=lw_q, alpha=alpha_q)
    inset.text(x_text, y_mean[0], 'InSAR', va='center', fontsize=7)
    inset.set_xlim(0, 2.2)
    inset.set_ylim(0.1, 0.9)
    
    
    
def plot_cores(imethods, fnout=None):
    from scripts.core_analysis import read_site, bootstrap_percentiles
    fig, axs = prepare_figure(
        nrows=2, ncols=4, figsize=(1.00, 0.85), sharex=True, sharey=True, left=0.12, top=0.89, right=0.99,
        bottom=0.12, wspace=0.40, hspace=0.23)
    method = 'supernatant'
    site_labels = {'HV': 'Happy Valley: upland', 'HVE': 'Happy Valley: toe slope'}
    version = '2022'
    fns = {
            'HVE': 'FSA_Dalton_HVE_2023_20250909.xlsx', 'HV': 'FSA_Dalton_HV_2023_20240201.xlsx'}
    y_f_insitu = {'HV': 48.7, 'HVE': 46.9}
    delta_y = 10
    geospatial = load_geospatial(p0, imethod)
    y_grid_im = np.arange(5) * delta_y
    c_is, c_im = colslist[2], colslist[0]
    alpha_q, lw_q = 0.5, 0.5
    alpha, lw, alpha_shade = 0.8, 1.1, 0.2
    for jsite, site in enumerate(('HV', 'HVE')):
        lonlat = lonlats[site]
        rc = geospatial.rowcol(lonlat, crs='epsg:4326')
        e_grid = read_site(paths['cores'] / fns[site], method, version, delta_y)
        e_q_mean = bootstrap_percentiles(e_grid, (10, 90))
        print(site, e_grid.shape[0])
        e_mean = np.nanmean(e_grid, axis=0)
        y_grid = np.arange(len(e_mean)) * delta_y
        for jim, _imethod in enumerate(imethods):
            ax = axs[jsite, jim]
            e_mean_site_im = load_results(p0, _imethod, 'e_mean_depth_mean')[rc[0], rc[1],:]
            yf_mean_site_im = load_results(p0, _imethod, 'yf_mean')[rc[0], rc[1], -1]
            e_quantile_site_im = load_results(p0, _imethod, 'e_mean_depth_quantile')[rc[0], rc[1], ...]
            rmse = np.sqrt(np.mean(
                [(e_mean_site_im[jdepth] - e_mean[jdepth])**2 for jdepth in range(len(y_grid_im))]))
            print('\t', _imethod, f'{rmse:.2f}')
            _plot_core(
                ax, y_grid, e_mean, e_quantile=e_q_mean, c=c_is, alpha=alpha, alpha_quantile=alpha_shade)
            _plot_core(ax, y_grid_im, e_quantile_site_im[..., 0], c=c_im, alpha=alpha_q, lw=lw_q)
            _plot_core(ax, y_grid_im, e_quantile_site_im[..., 1], c=c_im, alpha=alpha_q, lw=lw_q)
            _plot_core(ax, y_grid_im, e_mean_site_im, e_quantile=None, c=c_im, alpha=alpha)
            ax.axhline(y_f_insitu[site], c='#aaaaaa', lw=0.5)
        ax.text(-1.60, 1.05, site_labels[site], ha='center', va='baseline', transform=ax.transAxes)
    _add_legend_inset(axs[0, 0], c_is, c_im, alpha, alpha_q, alpha_shade, lw, lw_q)
    ax.set_ylim((57, 0))
    ax.set_xlim((-0.05, 0.65))
    from string import ascii_lowercase
    for jax, ax in enumerate(axs.flatten()):
        ax.text(0.030, 0.025, f'{ascii_lowercase[jax]})', ha='left', va='baseline', transform=ax.transAxes)
    axs[0, 0].text(1.04, 0.11, 'TD', c='#aaaaaa', transform=axs[0, 0].transAxes, ha='left', va='baseline')
    for jax, ax in enumerate(axs[:, 0]):
        ax.text(-0.48, 0.50, 'depth [cm]', rotation=90, va='center', ha='right', transform=ax.transAxes)
    for jax, ax in enumerate(axs[1, :]):
        ax.text(0.50, -0.32, '$e$ [-]', va='baseline', ha='center', transform=ax.transAxes)
    for jax, ax in enumerate(axs[0, :]):
        ax.text(
            0.50, 1.20, methodlabels[imethods[jax]], ha='center', va='baseline', c='k', 
            transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def plot_cores_IS(imethod='IS_full', fnout=None):
    from scripts.core_analysis import read_site, bootstrap_percentiles
    fig, axs = prepare_figure(
        nrows=1, ncols=2, figsize=(1.50, 0.85), sharex=True, sharey=True, left=0.10, top=0.89, right=0.92,
        bottom=0.14, wspace=0.20, hspace=0.23)
    method = 'supernatant'
    site_labels = {'HV': 'Happy Valley: upland', 'HVE': 'Happy Valley: toe slope'}
    version = '2022'
    fns = {
            'HVE': 'FSA_Dalton_HVE_2023_20250909.xlsx', 'HV': 'FSA_Dalton_HV_2023_20240201.xlsx'}
    y_f_insitu = {'HV': 48.7, 'HVE': 46.9}
    imethod = 'IS_full'
    geospatial = load_geospatial(p0, imethod)
    c_is, c_im = colslist[2], colslist[0]
    alpha_q, lw_q = 0.5, 0.5
    delta_y = 1.0
    alpha, lw, alpha_shade = 0.8, 1.1, 0.2
    for jsite, site in enumerate(('HV', 'HVE')):
        lonlat = lonlats[site]
        rc = geospatial.rowcol(lonlat, crs='epsg:4326')
        e_grid = read_site(paths['cores'] / fns[site], method, version, delta_y)
        e_q_mean = bootstrap_percentiles(e_grid, (10, 90))
        e_mean = np.nanmean(e_grid, axis=0)
        y_grid = np.arange(len(e_mean)) * delta_y
        ax = axs[jsite]
        e_mean_site_im = load_results(p0, imethod, 'e_mean')[rc[0], rc[1],:]
        yf_mean_site_im = load_results(p0, imethod, 'yf_mean')[rc[0], rc[1], -1]
        e_quantile_site_im = load_results(p0, imethod, 'e_quantile')[rc[0], rc[1], ...]
        y_grid_im = np.arange(len(e_mean_site_im)) * 0.2
        _plot_core(
            ax, y_grid, e_mean, e_quantile=e_q_mean, c=c_is, alpha=alpha, alpha_quantile=alpha_shade)
        _plot_core(ax, y_grid_im, e_quantile_site_im[..., 0], c=c_im, alpha=alpha_q, lw=lw_q)
        _plot_core(ax, y_grid_im, e_quantile_site_im[..., 1], c=c_im, alpha=alpha_q, lw=lw_q)
        _plot_core(ax, y_grid_im, e_mean_site_im, e_quantile=None, c=c_im, alpha=alpha)
        ax.axhline(yf_mean_site_im * 100, c='#aaaaaa', lw=0.5)
        ax.text(0.50, 1.06, site_labels[site], ha='center', va='baseline', transform=ax.transAxes)
    _add_legend_inset(axs[1], c_is, c_im, alpha, alpha_q, alpha_shade, lw, lw_q)
    ax.set_ylim((57, 0))
    ax.set_xlim((-0.01, 0.65))
    from string import ascii_lowercase

    axs[0].text(1.04, 0.16, 'TD', c='#aaaaaa', transform=axs[0].transAxes, ha='left', va='baseline')
    axs[0].text(-0.18, 0.50, 'depth [cm]', rotation=90, va='center', ha='right', transform=axs[0].transAxes)
    for jax, ax in enumerate(axs.flatten()):
        ax.text(0.030, 1.025, f'{ascii_lowercase[jax]})', ha='right', va='baseline', transform=ax.transAxes)
        ax.text(0.50, -0.15, '$e$ [-]', va='baseline', ha='center', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def plot_cores_IS_HV(imethod='IS_full', fnout=None):
    from scripts.core_analysis import read_site, bootstrap_percentiles
    fig, ax = prepare_figure(
        nrows=1, ncols=1, figsize=(1.00, 0.85), sharex=True, sharey=True, left=0.18, top=0.89, right=0.97,
        bottom=0.14, wspace=0.20, hspace=0.23)
    method = 'supernatant'
    site_labels = {'HV': 'Happy Valley', 'HVE': 'Happy Valley: toe slope'}
    version = '2022'
    fns = {
            'HVE': 'FSA_Dalton_HVE_2023_20250909.xlsx', 'HV': 'FSA_Dalton_HV_2023_20240201.xlsx'}
    y_f_insitu = {'HV': 48.7, 'HVE': 46.9}
    geospatial = load_geospatial(p0, imethod)
    c_is, c_im = colslist[2], colslist[0]
    alpha_q, lw_q = 0.5, 0.5
    delta_y = 1.0 if 'IS' in imethod else 10.0 
    alpha, lw, alpha_shade = 0.8, 1.1, 0.2
    site = 'HV'
    lonlat = lonlats[site]
    rc = geospatial.rowcol(lonlat, crs='epsg:4326')
    e_grid = read_site(paths['cores'] / fns[site], method, version, delta_y)
    e_q_mean = bootstrap_percentiles(e_grid, (10, 90))
    e_mean = np.nanmean(e_grid, axis=0)
    y_grid = np.arange(len(e_mean)) * delta_y
    _plot_core(
        ax, y_grid, e_mean, e_quantile=e_q_mean, c=c_is, alpha=alpha, alpha_quantile=alpha_shade)
    if 'IS' in imethod:
        e_mean_site_im = load_results(p0, imethod, 'e_mean')[rc[0], rc[1],:]
        yf_mean_site_im = load_results(p0, imethod, 'yf_mean')[rc[0], rc[1], -1]
        e_quantile_site_im = load_results(p0, imethod, 'e_quantile')[rc[0], rc[1], ...]
        y_grid_im = np.arange(len(e_mean_site_im)) * 0.2
        _plot_core(ax, y_grid_im, e_quantile_site_im[..., 0], c=c_im, alpha=alpha_q, lw=lw_q)
        _plot_core(ax, y_grid_im, e_quantile_site_im[..., 1], c=c_im, alpha=alpha_q, lw=lw_q)
        _plot_core(ax, y_grid_im, e_mean_site_im, e_quantile=None, c=c_im, alpha=alpha)
    elif 'GM' in imethod:
        e_mean_site_im = load_results(p0, imethod, 'e_mean_depth_mean')[rc[0], rc[1],:]
        yf_mean_site_im = load_results(p0, imethod, 'yf_mean')[rc[0], rc[1], -1]
        e_quantile_site_im = load_results(p0, imethod, 'e_mean_depth_quantile')[rc[0], rc[1], ...]
        y_grid_im = np.arange(len(e_mean_site_im)) * delta_y
        _plot_core(ax, y_grid_im, e_quantile_site_im[..., 0], c=c_im, alpha=alpha_q, lw=lw_q)
        _plot_core(ax, y_grid_im, e_quantile_site_im[..., 1], c=c_im, alpha=alpha_q, lw=lw_q)
        _plot_core(ax, y_grid_im, e_mean_site_im, e_quantile=None, c=c_im, alpha=alpha)        
    else:
        raise ValueError
    for ymin, ymax in ((0, 7),(44, 60)):
        ax.axhspan(ymin, ymax, color='#cccccc', zorder=10, alpha=0.3, linewidth=0)
    ax.text(0.98, 0.98, 'poorly constrained', va='top', ha='right', transform=ax.transAxes, c='#888888')
    # ax.axhline(yf_mean_site_im * 100, c='#aaaaaa', lw=0.5)
    ax.text(0.50, 1.06, site_labels[site], ha='center', va='baseline', transform=ax.transAxes)
    _add_legend_inset(
        ax, c_is, c_im, alpha, alpha_q, alpha_shade, lw, lw_q, bbox_to_anchor=(0.50, 0.50, 0.40, 0.30))
    ax.set_ylim((57, 0))
    ax.set_xlim((-0.01, 0.65))
    # ax.text(1.04, 0.16, 'TD', c='#aaaaaa', transform=ax.transAxes, ha='left', va='baseline')
    ax.text(-0.16, 0.50, 'depth [cm]', rotation=90, va='center', ha='right', transform=ax.transAxes)
    ax.text(0.50, -0.17, '$e$ [-]', va='baseline', ha='center', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

if __name__ == '__main__':
    year = 2023
    imethod = 'IS'
    rmethod = 'mintpy'
    p0 = paths['processed'] / 'dalton' / str(year) / 'ecotype'
    pfig = paths['figures'] / 'dalton'

    imethods = ('GM_K3', 'GM_K2', 'GM_K5', 'IS')
    imethods_syn = ('GM_K2', 'GM_K3', 'GM_K5', 'IS')
    # plot_results(p0, imethods, fnout=pfig / 'maps.pdf')
    # plot_results_quantile(p0, imethods, fnout=pfig / 'maps_quantile.pdf')
    # plot_comparison(p0, imethods[:-1], 'IS', fnout=pfig / 'comparison.pdf')
    # plot_metrics(imethods_syn, fnout=pfig / 'synthetic.pdf')
    # plot_cores(imethods, fnout=pfig / 'cores.pdf')    
    # plot_cores_IS(fnout = pfig / 'cores_IS.pdf')
    plot_cores_IS_HV(fnout = pfig / 'core_HV_Dave.pdf')

    # plot_cores_IS_HV(imethod='GM_K3', fnout = pfig / 'core_HV_Dave_GM.pdf')


