'''
Created on Jul 15, 2020

@author: simon
'''
import pandas as pd
import numpy as np
import os
import datetime
import copy

from simulation.toolik import load_forcing
from simulation.stefan import stefan_integral_balance
from simulation.stratigraphy import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple, 
    StefanStratigraphyConstantE)


class Predictor():
    def __init__(self):
        pass
    
    def predict(self, forcing, params, fields=None, geom=None, **kwargs):
        raise NotImplementedError
    
    def _project(self, y, geom, covariance=False):
        exponent = 2 if covariance else 1
        y_los = y * np.cos(geom['ia']) ** exponent
        return y_los
    
class StefanPredictor(Predictor):
    fieldsdef = ('e', 'depth', 'dy')
    
    def __init__(self, fields=None):
        if fields is None:
            self.fields = self.fieldsdef
        else:
            self.fields = fields

    def predict(self, forcing, params, fields=None, geom=None, **kwargs):
        # forcing: just dailytemp for Stefan
        # geom['ia']: incidence angle in rad
        stefandict = self._stefan_internal(forcing, params, fields=fields, **kwargs)
        if geom is not None:
            stefandict['s_los'] = self._project(stefandict['s'], geom, covariance=False)
        return stefandict
        
    def _stefan_internal(self, forcing, params, fields=None, **kwargs):
        if fields is None: fields = self.fields
        s, yf = stefan_integral_balance(forcing, params=params, **kwargs)
        stefandict = {'s': s, 'yf': yf}
        if fields is not None:
            stefandict.update({field: params[field] for field in fields})
        else:
            stefandict.update(params)
        return stefandict

class PredictionEnsemble():
    def __init__(self, strat, predictor, geom=None):
        # creates an internal copy of predictor
        self.strat = strat
        self.predictor = predictor
        self.geom = geom
        self.results = None
        
    def predict(self, forcing, force_bulk=False, **kwargs):
        self.results = {}
        if self.strat.Nbatch == 0:
            self.results = self.predictor.predict(
                forcing, self.strat.params(), geom=self.geom, **kwargs)
        else:
            for batch in range(self.strat.Nbatch):
                res_batch = self.predictor.predict(
                    forcing, self.strat.params(batch=batch), geom=self.geom, **kwargs)
                for k in res_batch:
                    if k in self.results:
                        if not np.isscalar(self.results[k]):
                            self.results[k] = np.concatenate(
                                (self.results[k], res_batch[k]), axis=0)
                    else:
                        self.results[k] = res_batch[k]

    def extract_predictions(
            self, indices, field='s_los', C=None, rng=None, reference_only=False):
        # nn interpolation from time steps to observation epochs 
        # indices: ind of time steps 
        # C not None: add measurement noise
        if not reference_only:
            s = self.results[field][:, indices[1:]]
        else:
            s = self.results[field]
        s -= self.results[field][:, indices[0]][:, np.newaxis]
        if C is not None:
            if rng is None:
                rng = np.random.default_rng(seed=1)
            assert C.shape[0] == s.shape[1]
            obs_noise = rng.multivariate_normal(
                np.zeros(s.shape[1]), C, size=(s.shape[0],))
            s += obs_noise
        return s

def test():
    df = load_forcing()
    d0 = '2019-05-15'
    d1 = '2019-09-15'
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0

    strat = StratigraphyMultiple(StefanStratigraphySmoothingSpline(N=25000), Nbatch=12)
#     params = strat.params()
#     print(params['e'].shape)

#     def fun_wrapped():
#         stratb.draw_stratigraphy()
#         paramsb = stratb.params()
#         stefan_integral_balance(dailytemp, params=paramsb, steps=1)

    from timeit import timeit
    fun_wrapped = lambda: stefan_stratigraphy(dailytemp, strat, force_bulk=False, steps=1)
    print(f'{timeit(fun_wrapped, number=1)}')
#     s, yf = stefan_integral_balance(dailytemp, params=params, steps=0)
#     s2, yf2 = stefan_integral_balance(dailytemp, params=params, steps=1)
#     print(np.percentile(yf2[:, -1], [10, 50, 90]))
#     print(np.percentile((yf - yf2)[:, -15], [10, 50, 90]))
#     effect of C is very small; affects timing slightly
#     stefandict = stefan_stratigraphy(dailytemp, strat, force_bulk=False, steps=1)
#     print(np.percentile(stefandict['yf'][:, -1], [10, 50, 90]))

def parse_dates(datestr, strp='%Y%m%d'):
    if isinstance(datestr, str):
        return datetime.datetime.strptime(datestr, strp)
    else:
        return [parse_dates(ds, strp=strp) for ds in datestr]

def simulation():
    strp = '%Y-%m-%d'
    stefanparams = {'steps': 1}
    fn = '/home/simon/Work/gie/processed/kivalina2019/timeseries/disp_polygons2.p'
    from InterferometricSpeckle.storage import load_object
    C_obs = load_object(fn)['C']
    datestr = ['2019-06-02', '2019-06-14', '2019-06-26', '2019-07-08', '2019-07-20',
               '2019-08-01', '2019-08-13', '2019-08-25', '2019-09-06']
    dates = parse_dates(datestr, strp=strp)
    geom = {'ia': 30 * np.pi / 180}
    df = load_forcing()
    d0 = '2019-05-28'
    d1 = '2019-09-15'
    d0_, d1_ = parse_dates((d0, d1), strp=strp)
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
#     dailytemp = 12 * np.sin((np.pi / len(dailytemp)) * np.arange(len(dailytemp)))
    ind_scenes = [int((d - d0_).days) for d in dates]

    predictor = StefanPredictor()
    strat = StratigraphyMultiple(StefanStratigraphySmoothingSpline(N=10000), Nbatch=3)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
#     strat_sim = StefanStratigraphySmoothingSpline(N=10, seed=114)  #122 #114
    strat_sim = StefanStratigraphyConstantE(N=10, seed=31)#29
    predens_sim = PredictionEnsemble(strat_sim, predictor, geom=geom)
    predens_sim.predict(dailytemp)
    
    rng = np.random.default_rng(seed=1)
    s_los_pred = predens.extract_predictions(ind_scenes, C=None, rng=rng)
    s_los_true = predens_sim.extract_predictions(ind_scenes, C=None, reference_only=True)

    # add loop over replicates here
    s_los_obs = predens_sim.extract_predictions(ind_scenes, C=C_obs, rng=rng)
    

    from inference import psislw, lw_mvnormal, expectation, sumlogs, quantile
    jsim = 0
    e_sim_jsim = predens_sim.results['e'][jsim, ...]
    s_los_obs_jsim = s_los_obs[jsim, ...]
    lw = lw_mvnormal(
        s_los_obs_jsim[np.newaxis, ...], C_obs[np.newaxis, ...], s_los_pred)
    lw_ps, _ = psislw(lw)
    e_inv = expectation(predens.results['e'], lw_ps, normalize=True)
    s_obs_inv = expectation(s_los_pred, lw_ps, normalize=True)
    yf_inv = expectation(predens.results['yf'], lw_ps, normalize=True)
    print(yf_inv[0, ind_scenes[-1]])
    e_inv_q = quantile(predens.results['e'], lw_ps, 0.1), quantile(predens.results['e'], lw_ps, 0.9)
     
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=2, sharey=False)
    fig.set_size_inches((8, 3), forward=True)
    days = np.arange(len(dailytemp))
    ygrid = np.arange(0, predens.results['depth'], step=predens.results['dy'])
    axs[0].plot(days[ind_scenes[1:]], s_los_obs_jsim, lw=0.0, c='k', alpha=0.6,
                marker='o', mfc='k', mec='none', ms=4)
    axs[0].plot(days, s_los_true[jsim, ...], lw=1.0, c='#999999', alpha=0.6)
#     axs[0].plot(days, s[ind_large[0, -1], :], lw=0.5, c='#ccccff', alpha=0.5)
#     axs[0].plot(days, s[ind_large[0, -2], :], lw=0.5, c='#ccccff', alpha=0.5)
#     axs[0].plot(days, s[ind_large[0, 9000], :], lw=0.5, c='#ffcccc', alpha=0.5)
    axs[1].plot(e_inv[0, :], ygrid, lw=1.0, c='#999999', alpha=0.6)
    axs[1].plot(e_sim_jsim, ygrid, lw=1.0, c='#000000', alpha=0.6)
    axs[1].plot(e_inv_q[0], ygrid, lw=0.5, c='#9999ee', alpha=0.6)
    axs[1].plot(e_inv_q[1], ygrid, lw=0.5, c='#9999ee', alpha=0.6)
    plt.show()


    # create separate simulation class with storage,  loop over 1000 samples
    # create separate class for simulation results, with distributed reading and comparison

if __name__ == '__main__':
    simulation()
