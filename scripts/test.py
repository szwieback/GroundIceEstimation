from analysis import (
    StefanPredictor, InversionSimulatorIS, InversionSimulatorGM, PredictionEnsemble, load_object, enforce_directory)
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple,
    StefanStratigraphyConstantE)
from scripts.pathnames import paths
from scripts.synthetic_simulation import sagwon_forcing, sagwon_covariance

import numpy as np

replicates = 4
N = 8192
Nsim = 5
Nbatch = 1
simname = 'test_spline'
inversion = 'gm'

fnforcing =  paths['forcing'] / 'sagwon/sagwon.csv'
# fnK = Path(f'/10TBstorage/Work/stacks/Dalton_131_363/gie/2019/proc/hadamard/geocoded/K_vec.geo.tif')
fnK = paths['stacks']/ 'Dalton_131_363/gie/2019/proc/hadamard/geocoded/K_vec.geo.tif'
pathout = paths['simulation'] / simname
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
ism = {'gm': InversionSimulatorGM, 'is': InversionSimulatorIS}[inversion]

fninvsim = pathout / 'invsim.p'
enforce_directory(fninvsim)

dailytemp, ind_scenes = sagwon_forcing(fnforcing)
variables = (('e', {'indranges': [(ind_scenes[-4], ind_scenes[-1])]}),)

C_obs = sagwon_covariance(fnK, var_atmo, wavelength=wavelength)
predictor = StefanPredictor()
strat = StratigraphyMultiple(
    StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)
strat_sim = StefanStratigraphySmoothingSpline(N=Nsim, seed=114)
predens_sim = PredictionEnsemble(strat_sim, predictor, geom=geom)
predens_sim.predict(dailytemp)
predens = PredictionEnsemble(strat, predictor, geom=geom)
predens.predict(dailytemp)
invsim = ism(predens=predens, predens_sim=predens_sim)
invsim.register_observations(ind_scenes, C_obs)
invsim.register_variables(variables)
invsim.export(fninvsim)

invsim = ism.from_file(fninvsim)
indranges = [(invsim.ind_scenes[-4], invsim.ind_scenes[-1])]
invsim.inference(replicates=replicates, pathout=pathout)
metrics = [('mean',), ('variance',)]
invsim.export_metrics(
    pathout, param=variables[0][0], indranges=variables[0][1]['indranges'], metrics_ind=metrics)


