'''
Created on Jul 15, 2020

@author: simon
'''
import pandas as pd
import numpy as np
import os
import datetime

from scripts.pathnames import paths
from analysis import (
    StefanPredictor, InversionSimulator, PredictionEnsemble, load_object, enforce_directory)
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple,
    StefanStratigraphyConstantE)


def toolik_simulation(
        simname, Nsim=500, replicates=250, N=25000, Nbatch=10, C_obs_multiplier=1.0):

    from forcing import read_toolik_forcing, parse_dates
    
    fn = os.path.join(paths['processed'], 'kivalina2019/timeseries/disp_polygons2.p')
    fnforcing = os.path.join(paths['forcing'], 'toolik2019', '1-hour_data.csv')
    pathout = os.path.join(paths['simulation'], simname)

    C_obs0 = load_object(fn)['C']
    var_atmo = (3e-3) ** 2
    C_obs0 += var_atmo * (np.ones_like(C_obs0) + np.eye(C_obs0.shape[0]))
    C_obs = C_obs0 * C_obs_multiplier
    datestr = ['2019-06-02', '2019-06-14', '2019-06-26', '2019-07-08', '2019-07-20',
               '2019-08-01', '2019-08-13', '2019-08-25', '2019-09-06']
    dates = parse_dates(datestr, strp='%Y-%m-%d')
    geom = {'ia': 30 * np.pi / 180}
    df = read_toolik_forcing(fnforcing)
    d0 = '2019-05-28'
    d1 = '2019-09-15'
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    ind_scenes = [int((d - d0_).days) for d in dates]

    predictor = StefanPredictor()

    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N), Nbatch=Nbatch)

    if simname.split('_')[0] in ['constant']:
        strat_sim = StefanStratigraphyConstantE(N=Nsim, seed=31)
    elif simname.split('_')[0] in ['spline']:
        strat_sim = StefanStratigraphySmoothingSpline(N=Nsim, seed=114)
    else:
        raise ValueError(f'Simulation {simname} not known')

    fninvsim = os.path.join(pathout, 'invsim.p')
    enforce_directory(fninvsim)

    predens_sim = PredictionEnsemble(strat_sim, predictor, geom=geom)
    predens_sim.predict(dailytemp)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    invsim = InversionSimulator(predens=predens, predens_sim=predens_sim)
    invsim.register_observations(ind_scenes, C_obs)
    invsim.export(fninvsim)

    invsim.logweights(replicates=replicates, pathout=pathout)
    invsim.export_metrics(pathout, param='e')
    invsim.export_metrics(pathout, param='e', prior=True)
    indranges = [(invsim.ind_scenes[-4], invsim.ind_scenes[-1])]
    invsim.export_metrics(pathout, param='e', indranges=indranges)
    invsim.export_metrics(pathout, param='e', indranges=indranges, prior=True)

def sagwon_forcing(fnforcing):
    from forcing import read_daily_noaa_forcing, parse_dates
    df = read_daily_noaa_forcing(fnforcing, convert_temperature=False)
    d0, d1 = '2019-05-11', '2019-09-17'
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())[pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    datesstr = ('20190521', '20190602', '20190614', '20190626', '20190708', '20190720',
                '20190801', '20190813', '20190825', '20190906')
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr]
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    return dailytemp, ind_scenes

def sagwon_covariance(fnK, var_atmo, wavelength=0.055, site=None, C_obs_multiplier=1.0):
    from analysis import read_K, add_atmospheric_K

    if site is None: site = np.array((-148.8317, 69.0414))[:, np.newaxis]

    K, geospatial_K = read_K(fnK)
    K = add_atmospheric_K(K, var_atmo, wavelength=wavelength)
    
    _rc_site = geospatial_K.rowcol(site)[:, 0]
    C_obs0 = K[..., _rc_site[0], _rc_site[1]]
    C_obs = C_obs0 * C_obs_multiplier

    return C_obs

def sagwon_simulation(
        simname, Nsim=500, replicates=250, N=25000, Nbatch=10, C_obs_multiplier=1.0):
    fnforcing = '/10TBstorage/Work/gie/forcing/sagwon/sagwon.csv'
    fnK = f'/10TBstorage/Work/stacks/Dalton_131_363/gie/2019/proc/hadamard/geocoded/K_vec.geo.tif'
    pathout = os.path.join(paths['simulation'], simname)
    params_distribution = {
        'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
        'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
        'wsat': {'low_above': 0.4, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},
        'soil': {'high_horizon': 0.20, 'low_horizon': 0.10, 'organic_above': 0.1,
                 'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
        'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}
    geom = {'ia': 40 * np.pi / 180}
    var_atmo = (4e-3) ** 2
    wavelength = 0.055

    dailytemp, ind_scenes = sagwon_forcing(fnforcing)
    
    C_obs = sagwon_covariance(
        fnK, var_atmo, wavelength=wavelength, C_obs_multiplier=C_obs_multiplier)

    predictor = StefanPredictor()

    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)

    if simname.split('_')[0] in ['constant']:
        strat_sim = StefanStratigraphyConstantE(N=Nsim, seed=31)
    elif simname.split('_')[0] in ['spline']:
        strat_sim = StefanStratigraphySmoothingSpline(N=Nsim, seed=114)
    else:
        raise ValueError(f'Simulation {simname} not known')

    fninvsim = os.path.join(pathout, 'invsim.p')
    enforce_directory(fninvsim)

    predens_sim = PredictionEnsemble(strat_sim, predictor, geom=geom)
    predens_sim.predict(dailytemp)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    invsim = InversionSimulator(predens=predens, predens_sim=predens_sim)
    invsim.register_observations(ind_scenes, C_obs)

    invsim.export(fninvsim)
    invsim.logweights(replicates=replicates, pathout=pathout)
    invsim.export_metrics(pathout, param='e')
    invsim.export_metrics(pathout, param='e', prior=True)
    indranges = [(invsim.ind_scenes[-4], invsim.ind_scenes[-1])]
    invsim.export_metrics(pathout, param='e', indranges=indranges)
    invsim.export_metrics(pathout, param='e', indranges=indranges, prior=True)


if __name__ == '__main__':
    N = 10000
    Nsim = 500
    replicates = 100
    multipliers = {'stdacc': 1.0, 'lowacc': 16.0, 'highacc': 1.0 / 16}
    Nbatch_list = [1, 10]
    # for Nbatch in Nbatch_list:
    #     for accn in multipliers:
    #         for scenarion in ['spline']:
    #             # toolik_simulation(
    #             #     f'{scenarion}_{accn}_{Nbatch}', N=N, Nsim=Nsim, replicates=replicates,
    #             #     Nbatch=Nbatch, C_obs_multiplier=multipliers[accn])
    #             sagwon_simulation(
    #                 f'{scenarion}_{accn}_{Nbatch}_sagwon', N=N, Nsim=Nsim, Nbatch=Nbatch, 
    #                 replicates=replicates, C_obs_multiplier=multipliers[accn])

    sagwon_simulation('spline_plot_sagwon', Nsim=100, N=N, replicates=5, Nbatch=1)
    