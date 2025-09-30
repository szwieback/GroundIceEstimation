'''
Created on Sep 29, 2025

@author: simon
'''
import numpy as np
from pathlib import Path
import shutil
import time

from analysis import (
    StefanPredictor, InversionSimulatorIS, InversionSimulatorGM, PredictionEnsemble, enforce_directory,
    load_object)
from simulation import (
    StefanStratigraphySmoothingSpline)

# representative parameters
params_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':-3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.6, 'high_above': 0.8, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.20, 'low_horizon': 0.10, 'organic_above': 0.1,
             'mineral_above': 0.00, 'mineral_below': 0.35, 'organic_below': 0.05},
    'n_factor': {'high': 1.00, 'low': 0.85, 'alphabeta': 2.0}}
geom = {'ia': 40 * np.pi / 180}
depthranges = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
variables = (('e', {'depthranges': depthranges}),)
inversion_types = ['GM_K2', 'IS']
metrics = [('mean',)]




def minimum_working_simulation(N=10000, Nsim=50000, pathres=None, delete=True):
    
    # If no path provided: create temporary directory
    if pathres is None:
        import tempfile
        pathres = Path(tempfile.mkdtemp())
        print(pathres)
    
    # Create fake data
    dailytemp = np.ones(95) * 15.0
    ind_scenes = np.arange(6, len(dailytemp), 14)
    C_obs = np.eye(len(ind_scenes) - 1)* 0.003
    
    # Set up predictions
    predictor = StefanPredictor()
    strat = StefanStratigraphySmoothingSpline(N=N, dist=params_distribution)
    strat_sim = StefanStratigraphySmoothingSpline(N=Nsim, seed=114)
    predens_sim = PredictionEnsemble(strat_sim, predictor, geom=geom)
    predens_sim.predict(dailytemp)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)    
    predens.predict_mean_depth(depthranges)

    # Loop over inversion methods
    for inversion in inversion_types:
        _simname = f'minimal_{inversion}'
        if inversion == 'IS':
            ism, kwargs = InversionSimulatorIS, {}
        else:
            assert inversion[0:2] == 'GM'
            K = int(inversion.split('_')[1][1:])
            ism, kwargs = InversionSimulatorGM, {'K': K}
            
        # Set up directory
        pathout = pathres / _simname
        fninvsim = pathout / 'invsim.p'
        enforce_directory(fninvsim)
    
        # Perform inversion
        start = time.perf_counter()
        invsim = ism(predens=predens, predens_sim=predens_sim, **kwargs)
        invsim.register_observations(ind_scenes, C_obs)
        invsim.register_variables(variables)
        invsim.inference(replicates=1, pathout=pathout, n_jobs=4)
        invsim.export_metrics(
            pathout, param='e', depthranges=depthranges, metrics_ind=metrics, metrics=[('RMSE',)])
        end = time.perf_counter()
        
        # Compute RMSE
        m = load_object(pathout / 'metrics_e_depthranges.p')
        med_rmse = np.median(m['RMSE'])
        
        # Print output
        print(f"Method: {inversion}")
        print(f"    Elapsed time: {end - start:.2f} seconds")
        print(f"    Median RMSE (excess ground ice): {med_rmse:.2f} [-]")

    if delete:
        shutil.rmtree(pathres)
    
if __name__ == '__main__':
    ### Run simulation
    N = int(1e4) # number of prior ensemble members
    Nsim = int(5e4) # number of simulations
    minimum_working_simulation(N=N, Nsim=Nsim, delete=True)
    
    