'''
Created on Sep 20, 2021

@author: simon
'''
import os
import matplotlib.pyplot as plt
import numpy as np
from pathnames import paths
from analysis.ioput import load_object
from analysis.synthetic import StefanPredictor, inversionSimulator, PredictionEnsemble

globfigparams = {
    'fontsize':8, 'family':'serif', 'usetex': True,
    'preamble': r'\usepackage{amsmath} \usepackage{times} \usepackage{mathtools}',
    'column_inch':229.8775 / 72.27, 'markersize':24, 'markercolour':'#AA00AA',
    'fontcolour':'#666666', 'tickdirection':'out', 'linewidth': 0.5,
    'ticklength': 2.50, 'minorticklength': 1.1 }
plt.rc(
    'font', **{'size':globfigparams['fontsize'], 'family':globfigparams['family']})
plt.rcParams['text.usetex'] = globfigparams['usetex']
plt.rcParams['text.latex.preamble'] = globfigparams['preamble']
plt.rcParams['legend.fontsize'] = globfigparams['fontsize']
plt.rcParams['font.size'] = globfigparams['fontsize']
plt.rcParams['axes.linewidth'] = globfigparams['linewidth']
plt.rcParams['axes.labelcolor'] = globfigparams['fontcolour']
plt.rcParams['axes.edgecolor'] = globfigparams['fontcolour']
plt.rcParams['xtick.color'] = globfigparams['fontcolour']
plt.rcParams['xtick.direction'] = globfigparams['tickdirection']
plt.rcParams['ytick.direction'] = globfigparams['tickdirection']
plt.rcParams['ytick.color'] = globfigparams['fontcolour']
plt.rcParams['xtick.major.width'] = globfigparams['linewidth']
plt.rcParams['ytick.major.width'] = globfigparams['linewidth']
plt.rcParams['xtick.minor.width'] = globfigparams['linewidth']
plt.rcParams['ytick.minor.width'] = globfigparams['linewidth']
plt.rcParams['ytick.major.size'] = globfigparams['ticklength']
plt.rcParams['xtick.major.size'] = globfigparams['ticklength']
plt.rcParams['ytick.minor.size'] = globfigparams['minorticklength']
plt.rcParams['xtick.minor.size'] = globfigparams['minorticklength']
plt.rcParams['text.color'] = globfigparams['fontcolour']
cols = {'true': '#000000', 'est': '#aa9966', 'unc': '#9999ee'}

def _plot_example(
        axs, sie, jsim=0, replicate=0, show_quantile=True, smooth_quantile=None, ymax=None,
        slim=None, sticks=None, show_ylabels=False):
    ygrid = sie.ygrid
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
    days = np.arange(s_sim.shape[1])
    d0 = sie.invsim.ind_scenes[0]
    axs[0].axhline(0.0, lw=0.5, c='#cccccc')
    axs[0].plot(
        days[sie.invsim.ind_scenes[1:]] - d0, s_obs[jsim, ...], lw=0.0, c='k',
        alpha=0.6, marker='o', mfc='k', mec='none', ms=4)
    axs[0].plot(
        days - d0, s_pred[jsim, ...] - s_pred[jsim, d0],
        c=cols['est'], lw=1.0)
    axs[0].plot(
        days - d0, s_sim[jsim, ...] - s_sim[jsim, d0],
        lw=1.0, c=cols['true'])
    if slim is not None:
        axs[0].set_ylim(slim)
    else:
        axs[0].set_ylim(list(axs[0].get_ylim())[::-1])
    if sticks is not None:
        axs[0].set_yticks(sticks)
    axs[1].plot(e_sim[jsim, :], ygrid, lw=1.0, c=cols['true'])
    alpha = sie.frac_thawed(replicate=replicate, jsim=jsim)
    for jdepth in np.arange(ygrid.shape[0] - 1):
        axs[1].plot(
            e_inv[jsim, jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=1.0,
            c=cols['est'], alpha=alpha[jdepth])
        if show_quantile:
            axs[1].plot(
                e_inv_q[jdepth:jdepth + 2, 0], ygrid[jdepth:jdepth + 2], lw=0.5,
                c=cols['unc'], alpha=alpha[jdepth])
            axs[1].plot(
                e_inv_q[jdepth:jdepth + 2, 1], ygrid[jdepth:jdepth + 2], lw=0.5,
                c=cols['unc'], alpha=alpha[jdepth])
        else:
            axs[1].plot(
                e_inv[jsim, jdepth:jdepth + 2] + e_inv_std[jsim, jdepth:jdepth + 2],
                 ygrid[jdepth:jdepth + 2], lw=0.5, c=cols['unc'], alpha=alpha[jdepth])
            axs[1].plot(
                e_inv[jsim, jdepth:jdepth + 2] - e_inv_std[jsim, jdepth:jdepth + 2],
                ygrid[jdepth:jdepth + 2], lw=0.5, c=cols['unc'], alpha=alpha[jdepth])
    if ymax is None: ymax = ygrid[-1]
    ylabxpos = -0.28
    axs[0].set_xlabel('time since first acquisition [d]')
    if show_ylabels:
        axs[0].text(
            ylabxpos, 0.5, 'subsidence [m]', transform=axs[0].transAxes, va='center',
            ha='right', rotation=90)
        axs[1].text(
            ylabxpos, 0.5, 'depth [m]', transform=axs[1].transAxes, va='center',
            ha='right', rotation=90)
    axs[1].set_ylim((ymax, ygrid[0]))

    axs[1].set_xlabel('excess ice content [-]')
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

def plot_examples(show_quantile=False):
    from collections import namedtuple
    simname = 'spline_plot'
    pathsim = os.path.join(paths['simulation'], simname)
    Instance = namedtuple('instance', ['replicate', 'jsim'])
    instances = (Instance(0, 18), Instance(0, 13), Instance(0, 12))  # 0, 2
    jsim = 18  # 13#12 #2
    ymax = 0.6
    slim = (0.075, -0.010)
    sticks = [0.0, 0.03, 0.06]

    fig, axs = plt.subplots(ncols=len(instances), nrows=2, sharey=False, sharex='row')
    plt.subplots_adjust(
        top=0.98, left=0.10, right=0.98, bottom=0.12, wspace=0.30, hspace=0.46)
    fig.set_size_inches((5, 3), forward=True)

    for jinstance, instance in enumerate(instances):
        invsim = inversionSimulator.from_file(os.path.join(pathsim, 'invsim.p'))
        sie = invsim.results(pathsim, replicates=(instance.replicate,))
        _plot_example(
            axs[:, jinstance], sie, jsim=instance.jsim, replicate=0,
            show_quantile=show_quantile, ymax=ymax, show_ylabels=(jinstance == 0),
            slim=slim, sticks=sticks)
    plt.savefig(os.path.join(paths['figures'], 'synthetic_examples.pdf'))

def plot_metrics(ymax=0.8):
    from string import ascii_lowercase
    import matplotlib.lines as mlines
    fig, axs = plt.subplots(ncols=3, sharey=True, sharex=False)
    plt.subplots_adjust(
        top=0.75, left=0.14, right=0.98, bottom=0.09, wspace=0.30, hspace=0.46)
    fig.set_size_inches((3.0, 2.2), forward=True)
    simnames = ['spline_highacc', 'spline_lowacc', 'spline_stdacc']
    colscen = {
        'spline_highacc':'#ad9e71', 'spline_lowacc':'#7171ae', 'spline_stdacc':'#4c4632'}
    alphascen = {'spline_highacc':0.5, 'spline_lowacc':0.5, 'spline_stdacc':0.8}
    lwscen = {'spline_highacc':0.6, 'spline_lowacc':0.6, 'spline_stdacc':1.2}
    labels = {'spline_highacc':'high', 'spline_lowacc':'low', 'spline_stdacc':'std'}

    axs[2].axvline(0.8, lw=0.5, c='#eeeeee')
    for simname in simnames:
        metrics = load_object(os.path.join(paths['simulation'], simname, 'metrics_e.p'))
        axs[0].plot(
            np.nanmean(metrics['MAD'], axis=0), metrics['ygrid'],
            lw=lwscen[simname], c=colscen[simname], alpha=alphascen[simname])
        axs[1].plot(
            np.nanmean(np.sqrt(metrics['variance']), axis=0), metrics['ygrid'],
            lw=lwscen[simname], c=colscen[simname], alpha=alphascen[simname])
        axs[2].plot(
            np.nanmean(metrics['coverage'][..., 1], axis=0), metrics['ygrid'],
            lw=lwscen[simname], c=colscen[simname], alpha=alphascen[simname])

    axs[0].set_xlim(0.05, 0.20)
    axs[1].set_xlim(0.00, 0.25)
    axs[2].set_xlim(0.35, 0.90)
    axs[2].set_xticks((0.4, 0.6, 0.8))
    axs[0].set_ylim(ymax, 0)
    axs[0].text(
        -0.4, 0.5, 'depth [m]', transform=axs[0].transAxes, va='center',
        ha='right', rotation=90)

    titles = ['error', 'posterior spread', 'coverage']
    xlabels = ['MAD [-]', '$\\sigma_p$ [-]', 'fraction [-]']
    for jax, ax in enumerate(axs):
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.text(
            0.54, 1.32, titles[jax], ha='center', va='baseline', c='k',
            transform=ax.transAxes)
        ax.text(
            0.54, 1.20, xlabels[jax], ha='center', va='baseline', transform=ax.transAxes)
        ax.text(
            0.05, 0.00, ascii_lowercase[jax] + ')', ha='left', va='baseline',
            transform=ax.transAxes)
    handles = []
    for simname in simnames:
        l = mlines.Line2D(
            [], [], lw=lwscen[simname], c=colscen[simname], alpha=alphascen[simname],
            label=labels[simname])
        handles.append(l)
    axs[1].legend(
        handles=[handles[-1]] + handles[:-1], loc='lower center', frameon=False,
        fancybox=False, ncol=3, bbox_to_anchor=(0.000, -0.196, 1.200, 0.100))
    axs[1].text(-1.480, -0.111, 'accuracy', transform=axs[1].transAxes)
    plt.savefig(os.path.join(paths['figures'], 'synthetic_metrics.pdf'))

def plot_metrics_presentation(simname, ymax=0.8):
    from string import ascii_lowercase
    import matplotlib.lines as mlines
    fig, axs = plt.subplots(ncols=3, sharey=True, sharex=False)
    plt.subplots_adjust(
        top=0.78, left=0.14, right=0.98, bottom=0.03, wspace=0.30, hspace=0.46)
    fig.set_size_inches((4.0, 2.2), forward=True)
    colscen = '#4c4632'
    alphascen = 0.8
    lwscen = 1.2

    axs[2].axvline(0.8, lw=0.5, c='#eeeeee')
    metrics = load_object(os.path.join(paths['simulation'], simname, 'metrics_e.p'))
    axs[0].plot(
        np.nanmean(metrics['MAD'], axis=0), metrics['ygrid'],
        lw=lwscen, alpha=alphascen, c=colscen)
    axs[1].plot(
        np.nanmean(np.sqrt(metrics['variance']), axis=0), metrics['ygrid'],
        lw=lwscen, alpha=alphascen, c=colscen)
    axs[2].plot(
        np.nanmean(metrics['coverage'][..., 1], axis=0), metrics['ygrid'],
        lw=lwscen, alpha=alphascen, c=colscen)

    axs[0].set_xlim(0.05, 0.20)
    axs[1].set_xlim(0.00, 0.25)
    axs[2].set_xlim(0.35, 0.90)
    axs[2].set_xticks((0.4, 0.6, 0.8))
    axs[0].set_ylim(ymax, 0)
    axs[0].text(
        -0.4, 0.5, 'depth [m]', transform=axs[0].transAxes, va='center',
        ha='right', rotation=90)

    titles = ['error', 'posterior spread', 'coverage']
    xlabels = ['MAD [-]', '$\\sigma_p$ [-]', 'fraction [-]']
    for jax, ax in enumerate(axs):
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.text(
            0.54, 1.24, titles[jax], ha='center', va='baseline', c='k',
            transform=ax.transAxes)
        ax.text(
            0.54, 1.15, xlabels[jax], ha='center', va='baseline', transform=ax.transAxes)

    plt.savefig(os.path.join(paths['figures'], 'synthetic_metrics_pres.pdf'))

def plot_metrics_indrange():
    from string import ascii_lowercase
    import matplotlib.transforms as transforms
    fig, axs = plt.subplots(ncols=2, sharey=True, sharex=False)
    plt.subplots_adjust(
        top=0.87, left=0.21, right=0.98, bottom=0.34, wspace=0.30, hspace=0.46)
    fig.set_size_inches((2.2, 0.9), forward=True)
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
    for jsimname, simname in enumerate(simnames):
        metrics = load_object(
            os.path.join(paths['simulation'], simname, 'metrics_e_indranges.p'))
        axs[0].plot(
            np.nanmean(metrics['MAD'], axis=0)[jindrange], jsimname,
            linestyle='none', mfc=colscen[simname], alpha=alphascen[simname], marker=marker,
            ms=ms, mec='none')
        axs[1].plot(
            np.nanmean(metrics['coverage'][..., 1], axis=0)[jindrange], jsimname,
            linestyle='none', mfc=colscen[simname], alpha=alphascen[simname], marker=marker,
            ms=ms, mec='none')

    axs[0].set_xlim(0.00, 0.15)
    axs[1].set_xlim(0.35, 0.90)
    axs[1].set_xticks((0.4, 0.6, 0.8))
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
    plt.savefig(os.path.join(paths['figures'], 'synthetic_metrics_indrange.pdf'))

if __name__ == '__main__':
    plot_metrics()
    plot_metrics_indrange()
