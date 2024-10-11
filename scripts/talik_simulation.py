from analysis import StefanPredictor, PredictionEnsemble, InversionSimulatorIS, enforce_directory
from simulation import StefanStratigraphySmoothingSplineTalik
from scripts.pathnames import paths
from scripts.synthetic_simulation import sagwon_covariance, sagwon_forcing

from pathlib import Path
import numpy as np
from copy import deepcopy

fnforcing = paths['forcing'] / 'sagwon/sagwon.csv'
dailytemp, ind_scenes = sagwon_forcing(fnforcing)
var_atmo = (4e-3) ** 2
wavelength = 0.055

def talik_simulation(N=16, seed=2):
    anc = {'kf': 1.9, 'Cf': 1.5e6, 'Tf':-0.2, 'Ct': 3.0e6}
    dist = {
        'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
        'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
        'wsat': {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
        'soil': {'high_horizon': 0.3, 'low_horizon': 0.1, 'organic_above': 0.1,
                 'mineral_above': 0.05, 'mineral_below': 0.3, 'organic_below': 0.05},
        'n_factor': {'high': 0.95, 'low': 0.85, 'alphabeta': 2.0},
        'talik': {'low_depth': 0.2, 'high_depth': 0.4, 'probability': 1.0, 'high_thickness': 0.5,
                  'low_thickness': 0.2, 'frozen_fraction': 0.1}}
    strat = StefanStratigraphySmoothingSplineTalik(seed=seed, N=N, dist=dist, ancillary=anc)
    strat.draw_stratigraphy()
    distf = deepcopy(dist)
    distf['talik']['frozen_fraction'] = 1.0
    stratf = StefanStratigraphySmoothingSplineTalik(seed=seed, N=N, dist=distf, ancillary=anc)

    predictor = StefanPredictor()
    geom = {'ia': 38.40 / 180 * np.pi}

    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    predensf = PredictionEnsemble(stratf, predictor, geom=geom)
    predensf.predict(dailytemp)

    distb = {
        'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
        'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
        'wsat': {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
        'soil': {'high_horizon': 0.3, 'low_horizon': 0.1, 'organic_above': 0.1,
                 'mineral_above': 0.05, 'mineral_below': 0.3, 'organic_below': 0.05},
        'n_factor': {'high': 0.95, 'low': 0.85, 'alphabeta': 2.0},
        'talik': {'low_depth': 0.2, 'high_depth': 0.4, 'probability': 0.0, 'high_thickness': 0.5,
                  'low_thickness': 0.2, 'frozen_fraction': 0.1}}
    stratb = StefanStratigraphySmoothingSplineTalik(seed=seed, N=N, dist=distb, ancillary=anc)
    stratb.draw_stratigraphy()
    predensb = PredictionEnsemble(stratb, predictor, geom=geom)
    predensb.predict(dailytemp)    

    predensdict = {'talik': predens, 'frozen': predensf, 'baseline': predensb}

    return predensdict

def talik_synthetic(
        simname, Nsim=64, replicates=16, N=50000):
    fnK = paths['stacks']/'Dalton_131_363/gie/2019/proc/hadamard/geocoded/K_vec.geo.tif'

    pathout = paths['simulation'] / simname

    C_obs = sagwon_covariance(fnK, var_atmo, wavelength=wavelength)

    fninvsim = pathout / 'invsim.p'
    enforce_directory(fninvsim)

    predens_dict = talik_simulation(N=N)
    predens_sim_dict = talik_simulation(N=Nsim, seed=654)
    if 'frozen' in simname:
        _predens = predens_dict['frozen']
    elif 'baseline' in simname:
        _predens = predens_dict['baseline']
    else:
        _predens = predens_dict['talik']
    invsim = InversionSimulatorIS(predens=_predens, predens_sim=predens_sim_dict['talik'])
    invsim.register_observations(ind_scenes, C_obs)

    invsim.export(fninvsim)
    invsim.inference(replicates=replicates, pathout=pathout)
    invsim.export_metrics(pathout, param='e')
    invsim.export_metrics(pathout, param='e', prior=True)
    indranges = [(invsim.ind_scenes[-3], invsim.ind_scenes[-1])]
    invsim.export_metrics(pathout, param='e', indranges=indranges)
    invsim.export_metrics(pathout, param='e', indranges=indranges, prior=True)

def plot_single(predens, predensf, fnout=None):
    from scripts.plotting import prepare_figure, colslist
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = prepare_figure(figsize=(1.0, 0.6))
    ind_ens = 1
    ax.plot(dailytemp.index, predens.results['yf'][ind_ens,:], alpha=0.5, c=colslist[0])
    ax.plot(dailytemp.index, predensf.results['yf'][ind_ens,:], alpha=0.5, c=colslist[2])
    ax.set_ylim((1.0, 0.0))
    ax.text(
        -0.18, 0.50, '$y_{\\mathrm{f}} [\\mathrm{m}]$', ha='left', va='center', transform=ax.transAxes,
        rotation=90)
    ax.text(0.80, 0.12, 'talik', c=colslist[0], transform=ax.transAxes, ha='left')
    ax.text(0.80, 0.54, 'frozen', c=colslist[2], transform=ax.transAxes, ha='left')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def plot_ensemble(predens, predensf, fnout=None):
    from scripts.plotting import prepare_figure, colslist
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    def pc(res):
        return np.nanpercentile(res, (25, 50, 75), axis=0)
    fig, axs = prepare_figure(nrows=2, figsize=(1.0, 0.7), right=0.9, bottom=0.09, wspace=0.05)
    yf, yff = pc(predens.results['yf']), pc(predensf.results['yf'])
    axs[0].fill_between(dailytemp.index, yf[0,:], yf[2,:], facecolor=colslist[0], alpha=0.1)
    axs[0].fill_between(dailytemp.index, yff[0,:], yff[2,:], facecolor=colslist[2], alpha=0.1)
    axs[0].plot(dailytemp.index, yf[1,:], alpha=0.5, c=colslist[0])
    axs[0].plot(dailytemp.index, yff[1,:], alpha=0.5, c=colslist[2])
    axs[0].set_ylim((1.0, 0.0))
    axs[0].text(
        -0.18, 0.50, '$y_{\\mathrm{f}} [\\mathrm{m}]$', ha='left', va='center', transform=axs[0].transAxes,
        rotation=90)
    axs[0].text(1.00, 0.10, 'talik', c=colslist[0], transform=axs[0].transAxes, ha='left')
    axs[0].text(1.00, 0.34, 'frozen', c=colslist[2], transform=axs[0].transAxes, ha='left')
    axs[0].set_yticks([0, 0.5, 1.0])

    sf, sff = pc(predens.results['s']), pc(predensf.results['s'])
    axs[1].fill_between(dailytemp.index, sf[0,:], sf[2,:], facecolor=colslist[0], alpha=0.1)
    axs[1].fill_between(dailytemp.index, sff[0,:], sff[2,:], facecolor=colslist[2], alpha=0.1)
    axs[1].plot(dailytemp.index, sf[1,:], alpha=0.5, c=colslist[0])
    axs[1].plot(dailytemp.index, sff[1,:], alpha=0.5, c=colslist[2])
    axs[1].set_ylim((0.2, 0.0))
    axs[1].set_yticks([0, 0.1, 0.2])
    axs[1].text(
        -0.18, 0.50, '$s [\\mathrm{m}]$', ha='left', va='center', transform=axs[1].transAxes,
        rotation=90)
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def plot_synthetic_twin(ymax=0.8, suffix=''):
    from string import ascii_lowercase
    from scripts.plotting import prepare_figure, colslist, cmap_e
    from analysis import load_object
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    fig, axs = prepare_figure(
        ncols=2, figsize=(0.95, 0.55), sharey=True, sharex=False,
        top=0.80, left=0.13, right=0.98, bottom=0.08, wspace=0.30, hspace=0.46,
        remove_spines=False)

    simnames = ('talik', 'talik_baseline')#, 'talik_frozen')
    colscen = {
        'talik_frozen':colslist[1], 'talik':colslist[0], 'talik_baseline': colslist[2], 'prior': colslist[3]}
    alphascen = {
        'talik_frozen':0.8,  'talik': 1.0, 'talik_baseline': 1.0, 'prior': 0.6}
    lwscen = {
        'talik_frozen':0.6,  'talik':1.2, 'talik_baseline': 0.6, 'prior': 0.3}
    labels = {
        'talik_frozen':'energy sink', 'talik':'include talik', 'talik_baseline': 'ignore talik'}

    for sim in simnames:
        simname = sim + suffix
        metrics = load_object(paths['simulation'] / simname / 'metrics_e.p')
        metrics_p = load_object(paths['simulation'] / simname / 'metrics_e_prior.p')
        axs[0].plot(
            np.nanmean(metrics['MAD'], axis=0), metrics['ygrid'],
            lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim])
        sharpness = np.nanmean(np.sqrt(metrics['variance']), axis=0)

        axs[1].plot(
            sharpness, metrics['ygrid'],
            lw=lwscen[sim], c=colscen[sim], alpha=alphascen[sim])
    axs[0].set_xlim(0.00, 0.21)
    axs[1].set_xlim(0.00, 0.32)  # 0.25
    axs[0].set_ylim(ymax, 0)
    axs[0].text(
        -0.25, 0.50, 'depth [cm]', transform=axs[0].transAxes, va='center',
        ha='right', rotation=90)

    xlabels = ['accuracy: MAD [-]', 'uncertainty: $\\sigma_{\\mathrm{p}}$ [-]']
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
    axs[0].legend(
        handles=handles, loc='lower center', frameon=False,
        fancybox=False, ncol=3, bbox_to_anchor=(0.600, -0.186, 1.200, 0.100),
        handlelength=1.0, handletextpad=0.5)
    plt.savefig(paths['figures'] / f'synthetic_talik.pdf')

def plot_synthetic_scatter_indrange(suffix='', subsample=1):
    from matplotlib import cm
    from scripts.plotting import prepare_figure, colslist, cmap_e
    from analysis import load_object
    from matplotlib.colors import Normalize
    import statsmodels.api as sm
    cmap = cmap_e
    # cmap = cc.cm['CET_CBL1']
    fig, ax = prepare_figure(
        ncols=1, sharey=True, sharex=False, figsize=(1.62, 0.90), figsizeunit='in',
        top=0.955, left=0.170, right=0.730, bottom=0.215, wspace=0.38, hspace=0.46, remove_spines=False)
    simname = f'talik{suffix}'
    pathsim = paths['simulation'] / simname
    metrics = load_object(pathsim / 'metrics_e_indranges.p')
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
    # ax.text(
    #     -0.520, ypos, 'a)', transform=ax.transAxes, ha='left', va='baseline')
    cax = fig.add_axes((0.80, 0.25, 0.05, 0.52))
    csmap = cm.ScalarMappable(norm=Normalize(vlim[0], vlim[1], clip=True), cmap=cmap)
    cbar = fig.colorbar(csmap , cax=cax, orientation='vertical')
    cbar.set_ticks(vlim)
    cbar.solids.set_rasterized(True)
    cbarlabel = 'KDE [$-$]'
    cax.text(1.0, 1.2, cbarlabel, ha='center', va='baseline', transform=cax.transAxes)
    fig.savefig(paths['figures'] / f'synthetic_talik_scatter_indrange{suffix}.pdf')

def plot_synthetic_metrics_indrange(suffix=''):
    import matplotlib.transforms as transforms
    from matplotlib import cm
    from scripts.plotting import prepare_figure, colslist, cmap_e
    from analysis import load_object
    from matplotlib.colors import Normalize    
    fig, axs = prepare_figure(
        ncols=3, sharey=True, sharex=False, figsize=(3.5, 0.9), figsizeunit='in',
        top=0.87, left=0.18, right=0.98, bottom=0.34, wspace=0.38, hspace=0.46)
    simnames = ('talik', 'talik_baseline')
    colscen = {
        'talik_baseline':colslist[2], 'talik':colslist[0], 'prior': colslist[3]}
    alphascen = {
        'talik_baseline':0.8, 'talik': 1.0, 'prior': 0.6}
    labels = {
        'talik_baseline':'ignore talik', 'talik':'include talik'}

    jindrange = 0
    marker = 'o'
    ms = 4
    colp, msp, mewp, alphap = '#999999', 3, 0.5, 0.4
    ylim = (1.5, -0.3)
    yticks = (0, 1)
    yticklabels = [labels[simname] for simname in simnames]
    def _sharpness(m):
        # s = np.nanmean(
        #     m['quantile'][..., 1] - m['quantile'][..., 0], axis=0) / 2
        s = np.nanmean(np.sqrt(m['variance']), axis=0)
        return s
    axs[2].axvline(80, lw=0.5, c='#eeeeee')
    for jsimname, sim in enumerate(simnames):
        simname = sim + suffix
        metrics = load_object(
            paths['simulation'] / simname / 'metrics_e_indranges.p')
        metrics_p = load_object(
            paths['simulation'] / simname / 'metrics_e_indranges_prior.p')
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
        0.90, 0.45, 'prior', rotation=270, ha='left', va='center', transform=axs[0].transAxes)
    axs[0].set_xlim(0.00, 0.16)
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
        # ax.text(
        #     0.03, 0.07, ascii_lowercase[jax + 1] + ')', ha='left', va='baseline',
        #     transform=ax.transAxes)
    axs[0].set_yticklabels(())
    trans = transforms.blended_transform_factory(
        axs[0].transAxes, axs[0].transData)
    for jtickl, tickl in enumerate(yticklabels):
        axs[0].text(xpos, jtickl, tickl, va='center', ha='right', transform=trans)
    fig.savefig(paths['figures'] / f'synthetic_talik_metrics_indrange{suffix}.pdf')

if __name__ == '__main__':
    pfig = paths['figures']
    # talik_synthetic('talik')
    # talik_synthetic('talik_frozen') # inversion ensemble frozen
    # talik_synthetic('talik_baseline') # inversion ensemble with baseline ice content

    # plot_single(predens, predensf, fnout=pfig / 'talik_single.pdf')
    # plot_ensemble(predens, predensf, fnout=pfig / 'talik_ens.pdf')

    # plot_synthetic_twin()
    # plot_synthetic_scatter_indrange()
    # plot_synthetic_scatter_indrange(suffix='_frozen')
    plot_synthetic_metrics_indrange()
