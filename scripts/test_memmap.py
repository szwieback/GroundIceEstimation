from analysis import (
    StefanPredictor, InversionSimulatorIS, InversionSimulatorGM, PredictionEnsemble, load_object,
    enforce_directory, InversionProcessorIS, InversionResultsIS, InversionResultsISMmap, InversionResults)
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple)
from scripts.pathnames import paths
from scripts.synthetic_simulation import sagwon_forcing, sagwon_covariance

import numpy as np

nrow, ncol = 8, 512
N = 128
Nbatch = 1
simname = 'test_spline'
inversion = 'is'

fnforcing = paths['forcing'] / 'sagwon/sagwon.csv'
# fnK = Path(f'/10TBstorage/Work/stacks/Dalton_131_363/gie/2019/proc/hadamard/geocoded/K_vec.geo.tif')
fnK = paths['stacks'] / 'Dalton_131_363/gie/2019/proc/hadamard/geocoded/K_vec.geo.tif'

pathout = paths['simulation'] / simname
# shutil.rmtree(pathout)
# pathout.mkdir()

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


dailytemp, ind_scenes = sagwon_forcing(fnforcing)
indranges = [(ind_scenes[-4], ind_scenes[-1])]
variables = (('e', {'indranges': indranges}),)

C_obs = sagwon_covariance(fnK, var_atmo, wavelength=wavelength)
predictor = StefanPredictor()
strat = StratigraphyMultiple(
    StefanStratigraphySmoothingSpline(N=N, dist=params_distribution), Nbatch=Nbatch)
strat_sim = StefanStratigraphySmoothingSpline(N=ncol, seed=114)
predens_sim = PredictionEnsemble(strat_sim, predictor, geom=geom)
predens_sim.predict(dailytemp)
predens = PredictionEnsemble(strat, predictor, geom=geom)
predens.predict(dailytemp)
predens.predict_mean_period(indranges)

IP = InversionProcessorIS
IR = InversionResults
ip = IP(predens, batch_size=ncol)

invsim = ism(predens=predens, predens_sim=predens_sim)
invsim.register_observations(ind_scenes, C_obs)
invsim.register_variables(variables)
ssim = invsim.simulated_observations().T
ssim = np.broadcast_to(ssim[..., None,:], (ssim.shape[0],) + (nrow, ncol,))
K = np.broadcast_to(C_obs[..., None, None], C_obs.shape + (nrow, ncol))

# ir = ip.results(
#         ind_scenes, ssim, K, pathout=pathout, n_jobs=-1, memory=False, overwrite=True)
# ir.save(pathout / 'ir.p')
ir = IR.from_file(pathout / 'ir.p')
# print(ir.memory, type(ir))
ir.blocksize = 1024

ir_mem = ip.results(
        ind_scenes, ssim, K, pathout=pathout, n_jobs=-1, memory=True, overwrite=False)
# print(np.allclose(ir.lw, ir_mem.lw))
# ip.delete_temporary(pathout)

expecs = [('yf', 'mean'), ('e', 'mean'), ('e', 'var'), ('e_mean_period', 'mean'),
          ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
          ('e_mean_period', 'quantile', {'quantiles': (0.1, 0.9)}),
          ('e', 'quantile', {'quantiles': (0.1, 0.9)})]
expec = expecs[1]
kwargs = expec[2] if len(expec) == 3 else {}
ir.register_dates(dailytemp.index)
ir.export_expectation(
    pathout, param=expec[0], etype=expec[1], hdf5=True, overwrite=True, **kwargs)
# e_var = ir.expectation(param='e', etype='var')
# print(e_var.shape)
# e_var_mem = ir_mem.expectation(param='e', etype='var')
# print(np.allclose(e_var, e_var_mem))

res_mem = ir_mem.expectation(param=expec[0], etype=expec[1], **kwargs)
res = np.load(pathout / f'{expec[0]}_{expec[1]}.npy', mmap_mode='r')
print(np.allclose(res_mem, res))
# then GM, including IR loading functionality (through _dict)
# then multiclass
# finally: always load IR in scripts

