'''
Created on Jul 25, 2022

@author: simon
'''
import os
import numpy as np
import pandas as pd
import copy

from scripts.synthetic_simulation import parse_dates
from scripts.pathnames import paths
from analysis import StefanPredictor, InversionSimulator, PredictionEnsemble, load_object
from simulation import (StefanStratigraphyPrescribedConstantE)

def toolik_sensitivity():
    from forcing import read_toolik_forcing

    fnforcing = os.path.join(paths['forcing'], 'toolik2019', '1-hour_data.csv')
    pathout = os.path.join(paths['simulation'], 'sensitivity')

    df = read_toolik_forcing(fnforcing)
    d0, d1 = '2019-05-28', '2019-09-15'
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0

    results = {}
    profiles = {}

    geom = {'ia': 0.0}
    e_sim = [0.05],# [0.01, 0.1]
    predictor = StefanPredictor()

    strat_sim = StefanStratigraphyPrescribedConstantE(N=len(e_sim))
    strat_sim.draw_stratigraphy()
    strat_sim.stratigraphy['e'][:, :] = np.array(e_sim)[:, np.newaxis]
    predens_sim = PredictionEnsemble(strat_sim, predictor, geom=geom)
    predens_sim.predict(dailytemp)
    results['baseline'] = predens_sim.results
    profiles['baseline'] = strat_sim.stratigraphy['e']
    ygrid = strat_sim._ygrid

    results['sensitivity'], profiles['sensitivity'] = [], []
    ind_dist = [75, 250]
    hind = 20
    y_dist = [ygrid[ind] for ind in ind_dist]
    print(y_dist)

    for ind in ind_dist:
        strat_sim_sens = copy.deepcopy(strat_sim)
        strat_sim_sens.stratigraphy['e'][:, ind - hind:ind + hind] += 0.1
        predens_sim_sens = PredictionEnsemble(strat_sim_sens, predictor, geom=geom)
        predens_sim_sens.predict(dailytemp)
        results['sensitivity'].append(predens_sim_sens.results)
        profiles['sensitivity'].append(strat_sim_sens.stratigraphy['e'])

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import datetime
    from scripts.plotting import prepare_figure, colslist
    ylims = [(0.05, 0.00),(0.8, 0.0),(0.015, -0.015)]
    yticks = [(0.00, 0.02, 0.04), (0.0, 0.3, 0.6), (-0.01, 0.00, 0.01)]
    yticklabels = [(0, 2, 4), (0, 30, 60), (-1, 0, 1)]
    ylabels = ['$s(t)$ [cm]', '$y_f(t)$ [cm]', '$\\delta y_f(t)$ [cm]']
    fig, axs = prepare_figure(
        ncols=len(e_sim), nrows=3, figsize=(0.7, 0.65), sharex='col', bottom=0.1, 
        left=0.17, right=0.98)
    _days = np.arange(results['baseline']['s'].shape[1])
    days = [d0_ + datetime.timedelta(days=int(d)) for d in _days]

    for jax, ax in enumerate(axs):
        ax.axhline(0.0, c='#dddddd', lw=0.4)
        ax.set_ylim(ylims[jax])
        ax.set_yticks(yticks[jax])
        ax.set_yticklabels(yticklabels[jax])
        ax.text(
            -0.13, 0.50, ylabels[jax], transform=ax.transAxes, ha='right', va='center', 
            rotation=90)
    
    for nsim in range(len(e_sim)):
        for jdist, ind in enumerate(ind_dist):
            c, lw = colslist[jdist + 1], 0.8
            axs[0].plot(days, results['sensitivity'][jdist]['s'][nsim, :], c=c, lw=lw)
            dyf = (results['sensitivity'][jdist]['yf'][nsim, :]
                    -results['baseline']['yf'][nsim, :])
            axs[1].axhspan(
                ygrid[ind - hind], ygrid[ind + hind], edgecolor='none', facecolor=c, 
                alpha=0.12)
            axs[2].plot(days, dyf, c=c, lw=lw)
#             ds = (results['sensitivity'][jdist]['s'][nsim, :]
#                   -results['baseline']['s'][nsim, :])
#             axs[4, nsim].plot(days[1:], np.diff(ds), c=c, lw=lw)
        c, lw = colslist[0], 1.1
        axs[0].plot(days, results['baseline']['s'][nsim, :], c=c, lw=lw)
        axs[1].plot(days, results['baseline']['yf'][nsim, :], c=c, lw=lw)
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axs[0].set_xlim((days[0], days[-1]))
    axs[1].text(0.60, 0.02, 'baseline', transform=axs[1].transAxes)
    axs[2].text(0.99, 0.11, 'shallow', transform=axs[2].transAxes, ha='right')
    axs[2].text(0.99, 0.89, 'deep', transform=axs[2].transAxes, ha='right')

    plt.show()

if __name__ == '__main__':
    toolik_sensitivity()

