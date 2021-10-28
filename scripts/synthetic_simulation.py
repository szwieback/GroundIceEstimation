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

def toolik_simulation(
        simname, Nsim=500, replicates=250, N=25000, Nbatch=10, C_obs_multiplier=1.0):
    fn = os.path.join(paths['processed'], 'kivalina2019/timeseries/disp_polygons2.p')
    pathout = os.path.join(paths['simulation'], simname)
    
    C_obs0 = load_object(fn)['C']
    C_obs0 += (4e-3) ** 2 * np.eye(C_obs0.shape[0])
    C_obs = C_obs0 * C_obs_multiplier
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

    if simname.split('_')[0] in ['constant']:
        strat_sim = StefanStratigraphyConstantE(N=Nsim, seed=31)
    elif simname.split('_')[0] in ['spline']:
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
    invsim.export_metrics(pathout, param='e')
    indranges = [(invsim.ind_scenes[-4], invsim.ind_scenes[-1])]
    invsim.export_metrics(pathout, param='e', indranges=indranges)


if __name__ == '__main__':
    Nsim = 500
    Nbatch = 10
    replicates = 100
    multipliers = {'lowacc': 16.0, 'highacc': 1.0/16}
    for accn in multipliers:
        for scenarion in ['spline']:
            toolik_simulation(
                f'{scenarion}_{accn}', Nsim=Nsim, replicates=replicates, Nbatch=Nbatch, 
                C_obs_multiplier=multipliers[accn])
#     
#     toolik_simulation('spline_plot', Nsim=50, replicates=5, Nbatch=4)
