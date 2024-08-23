from analysis import StefanPredictor, PredictionEnsemble
from simulation import StefanStratigraphySmoothingSplineTalik
from scripts.happyvalley import happyvalley_forcing

from pathlib import Path
import numpy as np
from copy import deepcopy

fnforcing = Path('/home/simon/Work/gie/forcing/sagwon/sagwon.csv')
dailytemp, ind_scenes = happyvalley_forcing(fnforcing, year=2019)


def talik_simulation(N = 16):
    anc= {'kf': 1.9, 'Cf': 1.5e6, 'Tf':-0.2, 'Ct': 3.0e6}

    dist = {
        'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
        'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
        'wsat': {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
        'soil': {'high_horizon': 0.3, 'low_horizon': 0.1, 'organic_above': 0.1,
                 'mineral_above': 0.05, 'mineral_below': 0.3, 'organic_below': 0.05},
        'n_factor': {'high': 0.95, 'low': 0.85, 'alphabeta': 2.0},
        'talik': {'low_depth': 0.2, 'high_depth': 0.4, 'probability': 1.0, 'high_thickness': 0.5,
                  'low_thickness': 0.2, 'frozen_fraction': 0.1}}
    strat = StefanStratigraphySmoothingSplineTalik(seed=2, N=N, dist=dist, ancillary=anc)
    strat.draw_stratigraphy()
    distf = deepcopy(dist)
    distf['talik']['frozen_fraction'] = 1.0
    stratf = StefanStratigraphySmoothingSplineTalik(seed=2, N=N, dist=distf, ancillary=anc)

    predictor = StefanPredictor()
    geom = {'ia': 38.40 / 180 * np.pi}
    
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    predensf = PredictionEnsemble(stratf, predictor, geom=geom)
    predensf.predict(dailytemp)    
    
    return predens, predensf
    
def plot_single(predens, predensf, fnout=None):
    from scripts.plotting import prepare_figure, colslist
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = prepare_figure(figsize=(1.0, 0.6))
    ind_ens = 1
    ax.plot(dailytemp.index, predens.results['yf'][ind_ens, :], alpha=0.5, c=colslist[0])
    ax.plot(dailytemp.index, predensf.results['yf'][ind_ens, :], alpha=0.5, c=colslist[2])
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
    axs[0].fill_between(dailytemp.index, yf[0, :], yf[2, :], facecolor=colslist[0], alpha=0.1)
    axs[0].fill_between(dailytemp.index, yff[0, :], yff[2, :], facecolor=colslist[2], alpha=0.1)    
    axs[0].plot(dailytemp.index, yf[1, :], alpha=0.5, c=colslist[0])
    axs[0].plot(dailytemp.index, yff[1, :], alpha=0.5, c=colslist[2])
    axs[0].set_ylim((1.0, 0.0))
    axs[0].text(
        -0.18, 0.50, '$y_{\\mathrm{f}} [\\mathrm{m}]$', ha='left', va='center', transform=axs[0].transAxes, 
        rotation=90)
    axs[0].text(1.00, 0.10, 'talik', c=colslist[0], transform=axs[0].transAxes, ha='left')
    axs[0].text(1.00, 0.34, 'frozen', c=colslist[2], transform=axs[0].transAxes, ha='left')
    axs[0].set_yticks([0, 0.5, 1.0])
    
    sf, sff = pc(predens.results['s']), pc(predensf.results['s'])
    axs[1].fill_between(dailytemp.index, sf[0, :], sf[2, :], facecolor=colslist[0], alpha=0.1)
    axs[1].fill_between(dailytemp.index, sff[0, :], sff[2, :], facecolor=colslist[2], alpha=0.1)    
    axs[1].plot(dailytemp.index, sf[1, :], alpha=0.5, c=colslist[0])
    axs[1].plot(dailytemp.index, sff[1, :], alpha=0.5, c=colslist[2])
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

if __name__ == '__main__':
    predens, predensf = talik_simulation()
    # plot_single(predens, predensf, fnout='/home/simon/Work/gie/figures/talik_single.pdf')
    plot_ensemble(predens, predensf, fnout='/home/simon/Work/gie/figures/talik_ens.pdf')
    