'''
Created on Sep 20, 2021

@author: simon
'''
import os
import matplotlib.pyplot as plt
import numpy as np

from scripts.pathnames import paths
from scripts.plotting import prepare_figure, colslist
from analysis import load_object
from analysis import InversionSimulator

cols = {'true': colslist[0], 'est': colslist[2], 'unc': colslist[2]}

def _plot_example(
        axs, sie, days=None, jsim=0, replicate=0, show_quantile=True, smooth_quantile=2,
        ymax=None, slim=None, sticks=None, show_ylabels=False):
    import matplotlib.dates as mdates
    ygrid = sie.ygrid
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
        days, conv * (s_pred[jsim, ...] - s_pred[jsim, d0]),
        c=cols['est'], lw=1.0)
    axs[0].plot(
        days, conv * (s_sim[jsim, ...] - s_sim[jsim, d0]),
        lw=1.0, c=cols['true'])
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
    alpha = sie.frac_thawed(replicate=replicate, jsim=jsim)
    for jdepth in np.arange(ygrid.shape[0] - 1):
        axs[1].plot(
            e_inv[jsim, jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=1.0,
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
            (e_inv - e_inv_std)[jsim, :], (e_inv + e_inv_std)[jsim, :],
            edgecolor='none', facecolor=cols['unc'], alpha=0.07)
    axs[1].plot(e_sim[jsim, :], ygrid, lw=1.0, c=cols['true'])
    if ymax is None: ymax = ygrid[-1]
    ylabxpos = -0.42
    if show_ylabels:
        axs[0].text(
            ylabxpos, 0.5, 'subsidence [cm]', transform=axs[0].transAxes, va='center',
            ha='right', rotation=90)
        axs[1].text(
            ylabxpos, 0.5, 'depth [m]', transform=axs[1].transAxes, va='center',
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
    from scripts.synthetic_simulation import parse_dates
    import matplotlib.lines as mlines
    import datetime
    simname = 'spline_plot'
    pathsim = os.path.join(paths['simulation'], simname)
    Instance = namedtuple('instance', ['replicate', 'jsim'])
    instances = (Instance(0, 2), Instance(0, 12) , Instance(0, 42))  # (0, 13) (0, 18)
    ymax = 0.7  # 0.6
    slim = (10, -1)
    sticks = [0, 3, 6, 9]

    d0, d1 = '2019-05-27', '2019-09-15'
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    _days = np.arange((d1_ - d0_).days)
    days = np.array([d0_ + datetime.timedelta(days=int(d)) for d in _days])
    fig, axs = prepare_figure(
        ncols=len(instances), nrows=2, sharey=False, sharex='row', figsize=(1.15, 0.7),
        top=0.98, left=0.13, right=0.98, bottom=0.14, wspace=0.30,
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
        handles, ('truth', 'estimate'), loc=3, frameon=False, ncol=1, 
        borderpad=0.00, handlelength=1.0, borderaxespad=0.3, handletextpad=0.6, 
        labelspacing=0.0, bbox_to_anchor=(0.06, 0.0, 0.5, 0.2))

    plt.savefig(os.path.join(paths['figures'], 'synthetic_examples.pdf'))

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
        'spline_stdacc':colslist[0]}
    alphascen = {'spline_highacc':0.5, 'spline_lowacc':0.5, 'spline_stdacc':0.8}
    lwscen = {'spline_highacc':0.6, 'spline_lowacc':0.6, 'spline_stdacc':1.2}
    labels = {'spline_highacc':'high', 'spline_lowacc':'low', 'spline_stdacc':'base'}

    axs[2].axvline(0.8, lw=0.5, c='#eeeeee')
    for sim in simnames:
        simname = sim + suffix
        metrics = load_object(os.path.join(paths['simulation'], simname, 'metrics_e.p'))
        metrics_p = load_object(
            os.path.join(paths['simulation'], simname, 'metrics_e_prior.p'))
#         axs[0].plot(np.nanmean(metrics_p['MAD'], axis=0), metrics['ygrid'])
        axs[0].plot(
            np.nanmean(metrics['MAD'], axis=0), metrics['ygrid'],
            lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim])
        
        sharpness = np.nanmean(np.sqrt(metrics['variance']), axis=0)
        sharpness = np.nanmean(
            metrics['quantile'][..., 1] - metrics['quantile'][..., 0], axis=0) / 2
        sharpness_p = np.nanmean(
            metrics_p['quantile'][..., 1] - metrics_p['quantile'][..., 0], axis=0) / 2
        axs[1].plot(sharpness_p, metrics['ygrid'])
        
        axs[1].plot(
            sharpness, metrics['ygrid'],
            lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim])
        axs[2].plot(
            np.nanmean(metrics['coverage'][..., 1], axis=0), metrics['ygrid'],
            lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim])

    axs[0].set_xlim(0.00, 0.20)
    axs[1].set_xlim(0.00, 0.30)#0.25
    axs[2].set_xlim(0.55, 0.95)
    axs[2].set_xticks((0.6, 0.8))
    axs[0].set_ylim(ymax, 0)
    axs[0].text(
        -0.4, 0.5, 'depth [m]', transform=axs[0].transAxes, va='center',
        ha='right', rotation=90)

    xlabels = ['accuracy MAD [-]', 'sharpness $\\sigma$ [-]', 'coverage [\%]']
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

def plot_metrics_indrange(suffix=''):
    from string import ascii_lowercase
    import matplotlib.transforms as transforms
    fig, axs = prepare_figure(
        ncols=2, sharey=True, sharex=False, figsize=(2.2, 0.9), figsizeunit='in',
        top=0.87, left=0.21, right=0.98, bottom=0.34, wspace=0.30, hspace=0.46)
    simnames = ['spline_lowacc', 'spline_stdacc', 'spline_highacc']
    colscen = {
        'spline_highacc':'#ad9e71', 'spline_lowacc':'#7171ae', 'spline_stdacc':'#4c4632'}
    alphascen = {'spline_highacc':0.5, 'spline_lowacc':0.5, 'spline_stdacc':0.8}

    jindrange = 0
    marker = 'o'
    ms = 4
    ylim = (-0.3, 2.5)
    yticks = (0, 1, 2)
    yticklabels = ('low', 'standard', 'high')

    axs[1].axvline(0.8, lw=0.5, c='#eeeeee')
    for jsimname, sim in enumerate(simnames):
        simname = sim + suffix
        metrics = load_object(
            os.path.join(paths['simulation'], simname, 'metrics_e_indranges.p'))
        axs[0].plot(
            np.nanmean(metrics['MAD'], axis=0)[jindrange], jsimname,
            linestyle='none', mfc=colscen[sim], alpha=alphascen[sim], marker=marker,
            ms=ms, mec='none')
        axs[1].plot(
            np.nanmean(metrics['coverage'][..., 1], axis=0)[jindrange], jsimname,
            linestyle='none', mfc=colscen[sim], alpha=alphascen[sim], marker=marker,
            ms=ms, mec='none')

    axs[0].set_xlim(0.00, 0.15)
    axs[1].set_xlim(0.55, 0.95)
    axs[1].set_xticks((0.6, 0.8))
    axs[0].set_ylim(ylim)
#
    titles = ['error', 'coverage']
    xlabels = ['MAD [-]', 'fraction [-]']
    ypos = 1.08
    xpos = -0.07
    for jax, ax in enumerate(axs):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks(yticks)
        ax.text(
            0.54, ypos, titles[jax], ha='center', va='baseline', c='k',
            transform=ax.transAxes)
        ax.text(
            0.54, -0.58, xlabels[jax], ha='center', va='baseline', transform=ax.transAxes)
        ax.text(
            0.03, 0.07, ascii_lowercase[jax] + ')', ha='left', va='baseline',
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
#     plot_examples(show_quantile=True)
#     for Nbatch in (1, 5, 10, 25):
#         plot_metrics(suffix=f'_{Nbatch}')
#         plot_metrics_indrange(suffix=f'_{Nbatch}')
    Nbatch = 1
    plot_metrics(suffix=f'_{Nbatch}')
