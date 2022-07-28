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
    d0, d1 = '2019-05-15', '2019-09-15'
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
    from scripts.plotting import prepare_figure, colslist
    fig, axs = prepare_figure(ncols=len(e_sim), nrows=4, sharex='row', sharey='row')
    days = np.arange(results['baseline']['s'].shape[1])
    for nsim in range(len(e_sim)):
        for jdist, ind in enumerate(ind_dist):
            c, lw = colslist[jdist + 1], 0.6
            axs[0].plot(days, results['sensitivity'][jdist]['s'][nsim, :], c=c, lw=lw)
            dyf = (results['sensitivity'][jdist]['yf'][nsim, :]
                    -results['baseline']['yf'][nsim, :])
            axs[2].plot(days, dyf, c=c, lw=lw)
            axs[3].plot(profiles['sensitivity'][jdist][nsim, :], ygrid, c=c, lw=lw)
#             ds = (results['sensitivity'][jdist]['s'][nsim, :]
#                   -results['baseline']['s'][nsim, :])
#             axs[4, nsim].plot(days[1:], np.diff(ds), c=c, lw=lw)
        c, lw = colslist[0], 1.2
        axs[0].plot(days, results['baseline']['s'][nsim, :], c=c, lw=lw)
        axs[1].plot(days, results['baseline']['yf'][nsim, :], c=c, lw=lw)
        axs[3].plot(profiles['baseline'][nsim, :], ygrid, c=c, lw=lw)

    plt.show()

if __name__ == '__main__':
    toolik_sensitivity()

