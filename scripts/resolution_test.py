import numpy as np

from scripts.pathnames import paths
from scripts.sensitivity import compute_sensitivity
from analysis import (
    StefanPredictor, InversionSimulator, PredictionEnsemble, enforce_directory, load_object, save_object)
from simulation import (StefanStratigraphySmoothingSpline, StratigraphyMultiple)

def resolution_scenario(fnout=None, overwrite=False):
    from scripts.synthetic_simulation import sagwon_covariance, sagwon_forcing
    fnforcing = paths['forcing'] / 'sagwon' / 'sagwon.csv'
    pathout = paths['simulation'] / 'resolution'
    fnK = paths['processed'] / 'Dalton_131_363' / '2019' / 'K_vec.geo.tif'
    dailytemp, ind_scenes = sagwon_forcing(fnforcing)
    # d0_ = dailytemp.index[0]
    geom = {'ia': 40 * np.pi / 180}
    var_atmo = (4e-3) ** 2
    wavelength = 0.055
    C_obs_multiplier = 1.0
    e_sim = [0.05],  # [0.01, 0.1]
    ind_dist = [100, 225]
    de, hind = 0.45, 25
    N, Nbatch, replicates = 10000, 25, 1

    results, meta = compute_sensitivity(e_sim, geom, dailytemp, ind_dist, de=de, hind=hind)
    params_distribution = {
        'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
        'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
        'wsat': {'low_above': 0.4, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},
        'soil': {'high_horizon': 0.20, 'low_horizon': 0.10, 'organic_above': 0.1,
                 'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
        'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}

    C_obs = sagwon_covariance(
        fnK, var_atmo, wavelength=wavelength, C_obs_multiplier=C_obs_multiplier)
    resolution_inversion(
        dailytemp, ind_scenes, C_obs, meta, geom, params_distribution, pathout, N=N, Nbatch=Nbatch,
        replicates=replicates, overwrite=True)
    plot_inversion(meta, pathout, fnout=fnout)

def plot_inversion(meta, pathout, N_ens=20, fnout=None):
    import matplotlib.pyplot as plt
    from scripts.plotting import prepare_figure, colslist
    from string import ascii_lowercase
    ymax = 60  # cm
    Nscen = len(meta['predens_sim'])
    fig, axs = prepare_figure(
        ncols=Nscen, nrows=1, sharey=True, sharex=True, figsize=(1.20, 0.65),
        top=0.89, left=0.105, right=0.990, bottom=0.145, wspace=0.30,
        hspace=0.35)
    ygrid = meta['predens_sim'][0].ygrid * 100  # cm
    dygrid = ygrid[1]
    alpha_ens, lw_ens = 0.3, 0.3
    alpha_pm, lw_pm = 1.0, 1.5
    scennames = ['baseline profile', 'enriched: shallow', 'enriched: deep']
    sl = [2.5, 5.5]
    alpham = 0.6
    cl = ['#777777', '#eeeeee']
    for jscen in range(Nscen):
        pathsim = pathout / str(jscen)
        invsim = InversionSimulator.from_file(pathsim / 'invsim.p')
        invres = invsim.results(pathsim)
        lw = invres.lw[0, 0,:]
        ind_ens = np.argsort(lw)[-N_ens:]
        e_ens = invres.predictions('e')[ind_ens,:]
        yff = invres.predictions('yf')[..., invres.ind_scenes[0]][ind_ens] * 100
        yfl = invres.predictions('yf')[..., invres.ind_scenes[-1]][ind_ens] * 100
        metrics = load_object(pathsim / 'metrics_e.p')
        e_sim = meta['predens_sim'][jscen].results['e'][0,:]
        axs[jscen].fill_betweenx(ygrid, 0 * e_sim, e_sim, ec='none', fc=colslist[jscen], alpha=0.2)
        axs[jscen].fill_betweenx(ygrid, 0 * e_sim, e_sim, fc='none', ec=colslist[jscen], lw=0.3)
        axs[jscen].plot(metrics['mean'][:, 0,:].T, ygrid, c='k', alpha=alpha_pm, lw=lw_pm)
        axs[jscen].plot(e_ens.T, ygrid, c='k', lw=lw_ens, alpha=alpha_ens)
        for n in range(N_ens):
            indf, indl = int(yff[n] / dygrid), int(yfl[n] / dygrid)
            axs[jscen].scatter(
                e_ens[n, indf], yff[n], marker='o', s=sl[1], c=cl[1], edgecolors='none', zorder=5, alpha=alpham)
            axs[jscen].scatter(
                e_ens[n, indf], yff[n], marker='o', s=sl[0], c=cl[0], edgecolors='none', zorder=5)
            axs[jscen].scatter(
                e_ens[n, indl], yfl[n], marker='o', s=sl[1], c=cl[0], edgecolors='none', zorder=5, alpha=alpham)
            axs[jscen].scatter(
                e_ens[n, indl], yfl[n], marker='o', s=sl[0], c=cl[1], edgecolors='none', zorder=5)
        axs[jscen].text(
            0.50, -0.18, '$e$ [$-$]', va='baseline', ha='center', transform=axs[jscen].transAxes)
        axs[jscen].text(
            0.01, 1.02, f"{ascii_lowercase[jscen + 3]}) {scennames[jscen]}", ha='left', va='baseline',
            transform=axs[jscen].transAxes)

    lax = axs[0].inset_axes([0.35, 0.03, 0.65, 0.27])
    lax.set_axis_off()
    y_l = [0.85, 0.60]
    kw = [{'lw': lw_pm, 'alpha': alpha_pm}, {'lw': lw_ens, 'alpha': alpha_ens}]
    labels_l = ['post. mean', 'ensemble']
    for _y, _l, _kw in zip(y_l, labels_l, kw):  #
        lax.plot((0.0, 0.2), (_y,) * 2, c='k', **_kw, transform=lax.transAxes)
        lax.text(0.4, _y, _l, va='center', ha='left', transform=lax.transAxes)
    y_l = [0.35, 0.10]
    _y = y_l[0]
    lax.scatter(0.1, _y, marker='o', s=sl[1], c=cl[1], edgecolors='none', zorder=5, transform=lax.transAxes)
    lax.scatter(0.1, _y, marker='o', s=sl[0], c=cl[0], edgecolors='none', zorder=5, transform=lax.transAxes, alpha=alpham)
    _y = y_l[1]
    lax.scatter(0.1, _y, marker='o', s=sl[1], c=cl[0], edgecolors='none', zorder=5, transform=lax.transAxes)
    lax.scatter(0.1, _y, marker='o', s=sl[0], c=cl[1], edgecolors='none', zorder=5, transform=lax.transAxes)
    for _y, label in zip(y_l, ['$y_{\\mathrm{f}}$ first', '$y_{\\mathrm{f}}$ last']):
        lax.text(0.4, _y, label, va='center', ha='left', transform=lax.transAxes)
    axs[0].set_ylim((ymax, ygrid[0]))
    axs[0].text(-0.27, 0.50, '$y$ [cm]', rotation=90, va='center', ha='right', transform=axs[0].transAxes)
    fig.text(0.5, 0.96, 'resolution analysis', c='k', ha='center', va='baseline', transform=fig.transFigure)
    if fnout is not None:
        fig.savefig(fnout)
    else:
        plt.show()

def resolution_inversion(
        dailytemp, ind_scenes, C_obs, meta, geom, params_distribution, pathout, N=10000, Nbatch=1,
        replicates=1, overwrite=False):
    predictor = StefanPredictor()
    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    Nscen = len(meta['predens_sim'])
    for jscen in range(Nscen):
        _pathout = pathout / str(jscen)
        fninvsim = _pathout / 'invsim.p'
        if overwrite or not fninvsim.exists():
            invsim = InversionSimulator(predens=predens, predens_sim=meta['predens_sim'][jscen])
            invsim.register_observations(ind_scenes, C_obs)
            enforce_directory(fninvsim)
            invsim.export(fninvsim)
            invsim.logweights(replicates=replicates, pathout=_pathout)
            invsim.export_metrics(_pathout, param='e')

if __name__ == '__main__':
    resolution_scenario(fnout=paths['figures'] / 'resolution.pdf')

