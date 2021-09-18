'''
Created on Jul 15, 2020

@author: simon
'''
import pandas as pd
import numpy as np
import os
import datetime

from analysis.ioput import load_object
from simulation.toolik import load_forcing
from pathnames import paths
from analysis.synthetic import StefanPredictor, inversionSimulator, PredictionEnsemble
from simulation.stratigraphy import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple,
    StefanStratigraphyConstantE)

def parse_dates(datestr, strp='%Y%m%d'):
    if isinstance(datestr, str):
        return datetime.datetime.strptime(datestr, strp)
    else:
        return [parse_dates(ds, strp=strp) for ds in datestr]

def toolik_simulation(simname, Nsim=500, replicates=250, N=25000, Nbatch=10):
    fn = os.path.join(paths['processed'], 'kivalina2019/timeseries/disp_polygons2.p')
    pathout = os.path.join(paths['simulation'], simname)

    C_obs = load_object(fn)['C']
    # weird correlation structure
    datestr = ['2019-06-02', '2019-06-14', '2019-06-26', '2019-07-08', '2019-07-20',
               '2019-08-01', '2019-08-13', '2019-08-25', '2019-09-06']
    dates = parse_dates(datestr, strp='%Y-%m-%d')
    geom = {'ia': 30 * np.pi / 180}
    df = load_forcing()
    d0 = '2019-05-28'
    d1 = '2019-09-15'
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
#     dailytemp = 12 * np.sin((np.pi / len(dailytemp)) * np.arange(len(dailytemp)))
    ind_scenes = [int((d - d0_).days) for d in dates]

    predictor = StefanPredictor()

    from simulation.stratigraphy import (
        StefanStratigraphyPrescribedConstantE, StefanStratigraphyPrescribedSmoothingSpline)
    strat = StratigraphyMultiple(
        StefanStratigraphyPrescribedSmoothingSpline(N=N), Nbatch=Nbatch)

    if simname in ['constant']:
        strat_sim = StefanStratigraphyConstantE(N=Nsim, seed=31)
    elif simname in ['spline']:
        strat_sim = StefanStratigraphySmoothingSpline(N=Nsim, seed=114)
    else:
        raise ValueError(f'Simulation {simname} not known')

    predens_sim = PredictionEnsemble(strat_sim, predictor, geom=geom)
    predens_sim.predict(dailytemp)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)

    invsim = inversionSimulator(predens=predens, predens_sim=predens_sim)
    invsim.register_observations(ind_scenes, C_obs)
    invsim.export(os.path.join(pathout, 'invsim.p'))
    invsim.logweights(replicates=replicates, pathout=pathout)
    invsim.export_metrics(pathout)

if __name__ == '__main__':
    Nsim = 250
    Nbatch = 4
    replicates = 100
    toolik_simulation('constant', Nsim=Nsim, replicates=replicates, Nbatch=Nbatch)
    toolik_simulation('spline', Nsim=Nsim, replicates=replicates, Nbatch=Nbatch)
    
#     res = load_object(os.path.join(paths['simulation'], 'constant', 'metrics.p'))
#     print(np.mean(res['RMSE'], axis=0))
#     print(np.mean(res['coverage'][..., 1], axis=0))

#     invsim_ = inversionSimulator.from_file(os.path.join(pathout, 'invsim.p'))
#     sie = invsim_.results(pathout)
#     jsim = 3
#     print(sie.expectation('e', replicate=replicate))
#     for replicate in range(replicates):
#         sie.plot(jsim=jsim, replicate=replicate, ymax=0.7, show_quantile=False)
#     sie.plot(jsim=jsim, replicate=3, ymax=0.7, show_quantile=True)
#     e_inv_q = sie.quantile(
#         [0.1], 'e', replicate=0, jsim=3, smooth=None)

#     indrange = (sie.invsim.ind_scenes[-4], sie.invsim.ind_scenes[-1])
#     e_mean = sie.mean_period(indrange)
#     e_mean_inv = sie.moment(param=None, replicate=None, p=e_mean)
#     np.set_printoptions(precision=3, suppress=True)
#     print(np.mean(e_mean_inv, axis=0))
#     print(np.std(e_mean_inv, axis=0))
#     print(np.mean(sie.invsim.predens_sim.results['e'], axis=1))

