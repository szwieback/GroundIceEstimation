'''
Created on Sep 20, 2021

@author: simon
'''
import os
import matplotlib.pyplot as plt
import numpy as np

from scripts.pathnames import paths
from scripts.plotting import prepare_figure, colslist, cmap_e
from analysis import load_object
from analysis import InversionSimulator

cols = {'est': colslist[2], 'true': colslist[0], 'unc': colslist[2]}

def _plot_example(
        axs, sie, days=None, jsim=0, replicate=0, show_quantile=True, smooth_quantile=2,
        ymax=None, slim=None, sticks=None, show_ylabels=False):
    import matplotlib.dates as mdates
    ygrid = sie.ygrid * 100  # cm
    conv = 100.0
    e_inv = sie.moment('e', replicate=replicate)
    e_inv_std = np.sqrt(sie.variance('e', replicate=replicate))
    if show_quantile:
        e_inv_q = sie.quantile(
            [0.1, 0.9], 'e', replicate=replicate, jsim=jsim,
            smooth=smooth_quantile, steps=10)
    e_sim = sie.prescribed('e')
    s_sim = sie.prescribed('s_los')
    s_obs = sie.observed(replicate=replicate)
    s_pred = sie.moment('s_los', replicate=replicate)
    if days is None: days = np.arange(s_sim.shape[1])
    d0 = sie.invsim.ind_scenes[0]
    axs[0].axhline(0.0, lw=0.4, c='#dddddd')
    axs[0].plot(
        days, conv * (s_pred[jsim, ...] - s_pred[jsim, d0]), c=cols['est'], lw=1.0, alpha=0.9)
    axs[0].plot(
        days, conv * (s_sim[jsim, ...] - s_sim[jsim, d0]), lw=1.0, c=cols['true'], alpha=0.9)
    ms, mew = 3, 0.5
    axs[0].plot(
        days[sie.invsim.ind_scenes[1:]], conv * s_obs[jsim, ...], lw=0.0, c=cols['est'],
        alpha=0.5, marker='o', mfc='w', mec='none', ms=ms, mew=mew)
    axs[0].plot(
        days[sie.invsim.ind_scenes[1:]], conv * s_obs[jsim, ...], lw=0.0, c=cols['est'],
        alpha=1.0, marker='o', mfc='none', mec=cols['est'], ms=ms, mew=mew)
    if slim is not None:
        axs[0].set_ylim(slim)
    else:
        axs[0].set_ylim(list(axs[0].get_ylim())[::-1])
    if sticks is not None:
        axs[0].set_yticks(sticks)
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axs[0].set_xlim((days[0], days[-1]))
    alpha = sie.frac_thawed(replicate=replicate, jsim=jsim) ** 3
    for jdepth in np.arange(ygrid.shape[0] - 1):
        axs[1].plot(
            e_inv[jsim, jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=0.6,
            c=cols['est'], alpha=alpha[jdepth])
        lw = 0.1
        if show_quantile:
            axs[1].plot(
                e_inv_q[jdepth:jdepth + 2, 0], ygrid[jdepth:jdepth + 2], lw=lw,
                c=cols['unc'], alpha=alpha[jdepth])
            axs[1].plot(
                e_inv_q[jdepth:jdepth + 2, 1], ygrid[jdepth:jdepth + 2], lw=lw,
                c=cols['unc'], alpha=alpha[jdepth])
        else:
            axs[1].plot(
                e_inv[jsim, jdepth:jdepth + 2] + e_inv_std[jsim, jdepth:jdepth + 2],
                 ygrid[jdepth:jdepth + 2], lw=lw, c=cols['unc'], alpha=alpha[jdepth])
            axs[1].plot(
                e_inv[jsim, jdepth:jdepth + 2] - e_inv_std[jsim, jdepth:jdepth + 2],
                ygrid[jdepth:jdepth + 2], lw=lw, c=cols['unc'], alpha=alpha[jdepth])
    if show_quantile:
        axs[1].fill_betweenx(
            ygrid,
            e_inv_q[:, 0], e_inv_q[:, 1],
            edgecolor='none', facecolor=cols['unc'], alpha=0.07)
    else:
        axs[1].fill_betweenx(
            ygrid,
            (e_inv - e_inv_std)[jsim,:], (e_inv + e_inv_std)[jsim,:],
            edgecolor='none', facecolor=cols['unc'], alpha=0.07)
    axs[1].plot(e_sim[jsim,:], ygrid, lw=1.0, c=cols['true'])
    if ymax is None: ymax = ygrid[-1]
    ylabxpos = -0.31
    if show_ylabels:
        axs[0].text(
            ylabxpos, 0.5, 'subsidence [cm]', transform=axs[0].transAxes, va='center',
            ha='right', rotation=90)
        axs[1].text(
            ylabxpos, 0.5, 'depth [cm]', transform=axs[1].transAxes, va='center',
            ha='right', rotation=90)
    else:
        axs[0].set_yticklabels([])
        axs[1].set_yticklabels([])
    axs[1].set_ylim((ymax, ygrid[0]))
    axs[1].text(
        0.5, -0.36, '$e$ [-]', transform=axs[1].transAxes, ha='center', va='baseline')
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

def plot_examples(show_quantile=False):
    from collections import namedtuple
    from forcing import parse_dates
    import matplotlib.lines as mlines
    import datetime
    from string import ascii_lowercase
    simname = 'spline_plot_sagwon'
    pathsim = os.path.join(paths['simulation'], simname)
    Instance = namedtuple('instance', ['replicate', 'jsim'])
    instances = (Instance(0, 54), Instance(1, 52), Instance(0, 95))
    labels = ('near-surface ice', 'ice poor', 'deep ice')
    
    ymax = 60
    slim = (10.1, -1.2)
    sticks = [0, 3, 6, 9]

    d0, d1 = '2019-05-11', '2019-09-18'
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    _days = np.arange((d1_ - d0_).days)
    days = np.array([d0_ + datetime.timedelta(days=int(d)) for d in _days])
    fig, axs = prepare_figure(
        ncols=len(instances), nrows=2, sharey=False, sharex='row', figsize=(1.10, 0.72),
        top=0.96, left=0.105, right=0.990, bottom=0.140, wspace=0.30,
        hspace=0.35)
    invsim = InversionSimulator.from_file(os.path.join(pathsim, 'invsim.p'))
    for jinstance, instance in enumerate(instances):
        axs[0, jinstance].text(
            0.500, 1.015, labels[jinstance], ha='center', va='baseline', 
            transform=axs[0, jinstance].transAxes)
        sie = invsim.results(pathsim, replicates=(instance.replicate,))
        _plot_example(
            axs[:, jinstance], sie, days=days, jsim=instance.jsim, replicate=0,
            show_quantile=show_quantile, ymax=ymax, show_ylabels=(jinstance == 0),
            slim=slim, sticks=sticks)
    handles = [
        mlines.Line2D([], [], color=col, lw=1) for col in (cols['true'], cols['est'])]
    for jax, ax in enumerate(axs.flatten()):
        ax.text(0.98, 0.04, f'{ascii_lowercase[jax]})', ha='right', va='baseline', transform=ax.transAxes)
    axs[0, 0].legend(
        handles, ('synthetic truth', 'estimate'), loc=3, frameon=False, ncol=1,
        borderpad=0.00, handlelength=0.8, borderaxespad=0.3, handletextpad=0.5,
        labelspacing=0.0, bbox_to_anchor=(0.05, 0.02, 0.5, 0.2))

    plt.savefig(os.path.join(paths['figures'], 'synthetic_examples_sagwon.pdf'))

def plot_examples_exploratory(show_quantile=False):
    from collections import namedtuple
    from forcing import parse_dates
    import matplotlib.lines as mlines
    import datetime
    simname = 'spline_plot_sagwon'
    pathsim = os.path.join(paths['simulation'], simname)
    Instance = namedtuple('instance', ['replicate', 'jsim'])
    k = 90
    # instances = (Instance(0, k), Instance(0, k+1) , Instance(0, k+2),
    #              Instance(0, k+3), Instance(0, k+4), Instance(0, k+5))
    instances = (Instance(0, 3), Instance(0, 34) , Instance(0, 51),
                 Instance(0, 52), Instance(0, 54), Instance(0, 95))
# 0, 3; #0, 34, 51, 52; 54
    ymax = 65
    slim = (11, -1)
    sticks = [0, 3, 6, 9]

    d0, d1 = '2019-05-11', '2019-09-18'
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    _days = np.arange((d1_ - d0_).days)
    days = np.array([d0_ + datetime.timedelta(days=int(d)) for d in _days])
    fig, axs = prepare_figure(
        ncols=len(instances), nrows=2, sharey=False, sharex='row', figsize=(2.10, 0.7),
        top=0.98, left=0.105, right=0.990, bottom=0.140, wspace=0.30,
        hspace=0.35)
    for jinstance, instance in enumerate(instances):
        invsim = InversionSimulator.from_file(os.path.join(pathsim, 'invsim.p'))
        sie = invsim.results(pathsim, replicates=(instance.replicate,))
        _plot_example(
            axs[:, jinstance], sie, days=days, jsim=instance.jsim, replicate=0,
            show_quantile=show_quantile, ymax=ymax, show_ylabels=(jinstance == 0),
            slim=slim, sticks=sticks)
    handles = [
        mlines.Line2D([], [], color=col, lw=1) for col in (cols['true'], cols['est'])]
    axs[0, 0].legend(
        handles, ('synth. truth', 'estimate'), loc=3, frameon=False, ncol=1,
        borderpad=0.00, handlelength=1.0, borderaxespad=0.3, handletextpad=0.6,
        labelspacing=0.0, bbox_to_anchor=(0.06, 0.0, 0.5, 0.2))

    plt.show()

def plot_metrics(ymax=0.8, suffix=''):
    from string import ascii_lowercase
    import matplotlib.lines as mlines
    fig, axs = prepare_figure(
        ncols=3, figsize=(0.95, 0.55), sharey=True, sharex=False,
        top=0.80, left=0.13, right=0.98, bottom=0.08, wspace=0.30, hspace=0.46,
        remove_spines=False)

    simnames = ('spline_highacc', 'spline_lowacc', 'spline_stdacc')
    colscen = {
        'spline_highacc':colslist[1], 'spline_lowacc':colslist[2],
        'spline_stdacc':colslist[0], 'prior': colslist[3]}
    alphascen = {
        'spline_highacc':0.8, 'spline_lowacc':0.8, 'spline_stdacc':1.0, 'prior': 0.6}
    lwscen = {
        'spline_highacc':0.6, 'spline_lowacc':0.6, 'spline_stdacc':1.2, 'prior': 0.3}
    labels = {
        'spline_highacc':'high', 'spline_lowacc':'low', 'spline_stdacc':'base'}

    axs[2].axvline(80, lw=0.5, c='#eeeeee')
    for sim in simnames:
        simname = sim + suffix
        metrics = load_object(os.path.join(paths['simulation'], simname, 'metrics_e.p'))
        metrics_p = load_object(
            os.path.join(paths['simulation'], simname, 'metrics_e_prior.p'))
        axs[0].plot(
            np.nanmean(metrics['MAD'], axis=0), metrics['ygrid'],
            lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim])
        # sharpness = np.nanmean(np.sqrt(metrics['variance']), axis=0)
        sharpness = np.nanmean(
            metrics['quantile'][..., 1] - metrics['quantile'][..., 0], axis=0) / 2
        sharpness_p = np.nanmean(
            metrics_p['quantile'][..., 1] - metrics_p['quantile'][..., 0], axis=0) / 2
        axs[1].plot(
            sharpness, metrics['ygrid'],
            lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim])
        axs[2].plot(
            np.nanmean(metrics['coverage'][..., 1], axis=0) * 100, metrics['ygrid'],
            lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim])

    axs[0].plot(
        np.nanmean(metrics_p['MAD'], axis=0), metrics['ygrid'], lw=lwscen['prior'],
        c=colscen['prior'], alpha=alphascen['prior'])
    axs[1].plot(sharpness_p, metrics['ygrid'], lw=lwscen['prior'],
        c=colscen['prior'], alpha=alphascen['prior'])
    axs[0].text(
        1.10, 0.12, 'prior', rotation=270, color=colscen['prior'], alpha=alphascen['prior'],
        transform=axs[0].transAxes, va='center', ha='right')
    axs[0].set_xlim(0.00, 0.21)
    axs[1].set_xlim(0.00, 0.32)  # 0.25
    axs[2].set_xlim(55, 95)
    axs[2].set_xticks((60, 80))
    axs[0].set_ylim(ymax, 0)
    axs[0].text(
        -0.4, 0.5, 'depth [cm]', transform=axs[0].transAxes, va='center',
        ha='right', rotation=90)

    xlabels = ['MAD [-]', 'uncertainty [-]', 'coverage [\%]']
    for jax, ax in enumerate(axs):
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.text(
            0.54, 1.20, xlabels[jax], ha='center', va='baseline', transform=ax.transAxes)
        ax.text(
            0.05, 0.00, ascii_lowercase[jax] + ')', ha='left', va='baseline',
            transform=ax.transAxes)
    handles = []
    for sim in simnames:
        l = mlines.Line2D(
            [], [], lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim],
            label=labels[sim])
        handles.append(l)
    axs[1].legend(
        handles=[handles[-1]] + handles[:-1], loc='lower center', frameon=False,
        fancybox=False, ncol=3, bbox_to_anchor=(0.200, -0.186, 1.200, 0.100),
        handlelength=1.0, handletextpad=0.5)
    axs[1].text(-1.10, -0.09, 'accuracy', transform=axs[1].transAxes)
    plt.savefig(os.path.join(paths['figures'], f'synthetic_metrics{suffix}.pdf'))

def plot_scatter_indrange(suffix='', subsample=100):
    import colorcet as cc
    from matplotlib import cm
    from matplotlib.colors import Normalize
    import statsmodels.api as sm
    cmap = cmap_e
    # cmap = cc.cm['CET_CBL1']
    fig, ax = prepare_figure(
        ncols=1, sharey=True, sharex=False, figsize=(1.62, 0.90), figsizeunit='in',
        top=0.955, left=0.170, right=0.730, bottom=0.215, wspace=0.38, hspace=0.46, remove_spines=False)
    simname = f'spline_stdacc{suffix}'
    pathsim = os.path.join(paths['simulation'], simname)
    metrics = load_object(os.path.join(pathsim, 'metrics_e_indranges.p'))
    post_mean = metrics['mean']
    presc = np.ones_like(post_mean)
    presc[:, ...] = metrics['sim'][np.newaxis, ...]
    values = np.vstack((presc.ravel(), post_mean.ravel()))
    values = values[:,::subsample]
    gridparms = (0, 0.9, 100)
    xx, yy = np.mgrid[gridparms[0]:gridparms[1]:gridparms[2] * 1j, gridparms[0]:gridparms[1]:gridparms[2] * 1j]
    positions = np.vstack((xx.ravel(), yy.ravel()))
    dens_c = sm.nonparametric.KDEMultivariateConditional(
        endog=[values[1,:]], exog=[values[0,:]], dep_type='c', indep_type='c', bw=(0.05,) * 2)
    vlim = (0.00, 5.00)
    f = dens_c.pdf(positions[1,:], positions[0,:]).reshape(xx.shape)
    ax.imshow(
        f, extent=(gridparms[0], gridparms[1]) * 2, cmap=cmap, origin='lower', vmin=vlim[0], vmax=vlim[1])
    ax.set_aspect('equal')
    ticks = (0.0, 0.3, 0.6, 0.9)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.text(
        -0.395, 0.500, '$\\hat{\\bar{e}}$ [$-$]', rotation=90, transform=ax.transAxes, ha='right',
        va='center')
    ypos = -0.235
    ax.text(
        1.270, ypos, '$\\bar{e}$ [$-$]', transform=ax.transAxes, ha='left', va='baseline')
    ax.text(
        -0.520, ypos, 'a)', transform=ax.transAxes, ha='left', va='baseline')
    cax = fig.add_axes((0.80, 0.25, 0.05, 0.52))
    csmap = cm.ScalarMappable(norm=Normalize(vlim[0], vlim[1], clip=True), cmap=cmap)
    cbar = fig.colorbar(csmap , cax=cax, orientation='vertical')
    cbar.set_ticks(vlim)
    cbar.solids.set_rasterized(True)
    cbarlabel = 'KDE [$-$]'
    cax.text(1.0, 1.2, cbarlabel, ha='center', va='baseline', transform=cax.transAxes)
    fig.savefig(os.path.join(paths['figures'], f'synthetic_scatter_indrange{suffix}.pdf'))

def plot_metrics_indrange(suffix=''):
    from string import ascii_lowercase
    import matplotlib.transforms as transforms
    fig, axs = prepare_figure(
        ncols=3, sharey=True, sharex=False, figsize=(2.5, 0.9), figsizeunit='in',
        top=0.87, left=0.18, right=0.98, bottom=0.34, wspace=0.38, hspace=0.46)
    simnames = ['spline_lowacc', 'spline_stdacc', 'spline_highacc']
    colscen = {
        'spline_highacc':'#ad9e71', 'spline_lowacc':'#7171ae', 'spline_stdacc':'#4c4632'}
    colscen = {
        'spline_highacc':colslist[2], 'spline_lowacc':colslist[1], 'spline_stdacc': colslist[0]}
    alphascen = {'spline_highacc':1.0, 'spline_lowacc':1.0, 'spline_stdacc':1.0}

    jindrange = 0
    marker = 'o'
    ms = 4
    colp, msp, mewp, alphap = '#999999', 3, 0.5, 0.4
    ylim = (-0.3, 2.5)
    yticks = (0, 1, 2)
    yticklabels = ('low', 'standard', 'high')
    def _sharpness(m):
        # s = np.nanmean(
        #     m['quantile'][..., 1] - m['quantile'][..., 0], axis=0) / 2
        s = np.nanmean(np.sqrt(m['variance']), axis=0)
        return s
    axs[2].axvline(80, lw=0.5, c='#eeeeee')
    for jsimname, sim in enumerate(simnames):
        simname = sim + suffix
        metrics = load_object(
            os.path.join(paths['simulation'], simname, 'metrics_e_indranges.p'))
        metrics_p = load_object(
            os.path.join(paths['simulation'], simname, 'metrics_e_indranges_prior.p'))
        axs[0].plot(
            np.nanmean(metrics['MAD'], axis=0)[jindrange], jsimname,
            linestyle='none', mfc=colscen[sim], alpha=alphascen[sim], marker=marker,
            ms=ms, mec='none')
        axs[0].plot(
            np.nanmean(metrics_p['MAD'], axis=0)[jindrange], jsimname,
            linestyle='none', mec=colp, alpha=alphap, marker=marker,
            mew=mewp, ms=msp, mfc='none')
        axs[1].plot(
            _sharpness(metrics), jsimname, linestyle='none', mfc=colscen[sim], alpha=alphascen[sim],
            marker=marker, ms=ms, mec='none')
        axs[1].plot(
            _sharpness(metrics_p), jsimname, linestyle='none', mec=colp, alpha=alphap, marker=marker,
            mew=mewp, ms=msp, mfc='none')
        axs[2].plot(
            100 * np.nanmean(metrics['coverage'][..., 1], axis=0)[jindrange], jsimname,
            linestyle='none', mfc=colscen[sim], alpha=alphascen[sim], marker=marker,
            ms=ms, mec='none')
    axs[0].text(
        0.99, 0.45, 'prior', rotation=270, ha='left', va='center', transform=axs[0].transAxes)
    axs[0].set_xlim(0.00, 0.23)
    axs[0].set_xticks((0.00, 0.10, 0.20))
    axs[1].set_xlim(0.00, 0.35)
    axs[1].set_xticks((0.00, 0.20))
    axs[2].set_xlim(55, 95)
    axs[2].set_xticks((60, 80))
    axs[0].set_ylim(ylim)

    xlabels = ['MAD [$-$]', '$\\sigma_{\\mathrm{p}}$ [$-$]', 'coverage [\%]']
    ypos = 1.08
    xpos = -0.07
    for jax, ax in enumerate(axs):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks(yticks)
        # ax.text(
        #     0.54, ypos, titles[jax], ha='center', va='baseline', c='k',
        #     transform=ax.transAxes)
        ax.text(
            0.540, -0.575, xlabels[jax], ha='center', va='baseline', transform=ax.transAxes)
        ax.text(
            0.03, 0.07, ascii_lowercase[jax + 1] + ')', ha='left', va='baseline',
            transform=ax.transAxes)
    axs[0].text(
        xpos, ypos, 'accuracy', va='baseline', ha='right', transform=axs[0].transAxes)
    axs[0].set_yticklabels(())
    trans = transforms.blended_transform_factory(
        axs[0].transAxes, axs[0].transData)
    for jtickl, tickl in enumerate(yticklabels):
        axs[0].text(xpos, jtickl, tickl, va='center', ha='right', transform=trans)
    plt.savefig(os.path.join(paths['figures'], f'synthetic_metrics_indrange{suffix}.pdf'))

if __name__ == '__main__':
    # plot_examples(show_quantile=True)
    # plot_examples_exploratory(show_quantile=False)
    # plot_metrics_indrange(suffix=f'_1_sagwon_indrange')
    # plot_scatter_indrange(suffix=f'_1_sagwon_indrange', subsample=1)
    for Nbatch in (1, 10,):
        plot_metrics(suffix=f'_{Nbatch}_sagwon')
    #     # plot_metrics_indrange(suffix=f'_{Nbatch}')

