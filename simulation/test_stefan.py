'''
Created on Jul 15, 2020

@author: simon
'''
import pandas as pd
import numpy as np
import os
import datetime

from simulation.toolik import load_forcing
from simulation.stefan import stefan_ens, stefan_integral_balance
from simulation.stefan_single import stefan_integral_balance_single
from simulation.stratigraphy import (
    StefanStratigraphy, StefanStratigraphySmoothingSpline, StratigraphyMultiple, 
    StefanStratigraphyConstantE)

fieldsdef = ('e', 'depth', 'dy')

def stefan_stratigraphy(dailytemp, strat, fields=fieldsdef, force_bulk=False, **kwargs):
    def _stefan_internal(params):
        s, yf = stefan_integral_balance(dailytemp, params=params, **kwargs)
        stefandict = {'s': s, 'yf': yf}
        if fields is not None:
            stefandict.update({field: params[field] for field in fields})
        else:
            stefandict.update(params)
        return stefandict
    if strat.Nbatch == 0 or force_bulk:
        params = strat.params()
        stefandict = _stefan_internal(params)
    else:
        for batch in range(strat.Nbatch):
            params_batch = strat.params(batch=batch)
            stefandict_batch = _stefan_internal(params_batch)
            if batch == 0:
                stefandict = stefandict_batch
            else:
                for k in stefandict.keys():
                    if not np.isscalar(stefandict[k]):
                        stefandict[k] = np.concatenate(
                            (stefandict[k], stefandict_batch[k]), axis=0)
    return stefandict

def test():
    df = load_forcing()
    d0 = '2019-05-15'
    d1 = '2019-09-15'
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0

    stratb = StefanStratigraphySmoothingSpline(N=300000)

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
    res = load_object(fn)
    datestr = ['2019-06-02', '2019-06-14', '2019-06-26', '2019-07-08', '2019-07-20',
               '2019-08-01', '2019-08-13', '2019-08-25', '2019-09-06']
    dates = parse_dates(datestr, strp=strp)
    ia = 30 * np.pi / 180
    df = load_forcing()
    d0 = '2019-05-28'
    d1 = '2019-09-15'
    d0_, d1_ = parse_dates((d0, d1), strp=strp)
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    dailytemp = 12 * np.sin((np.pi / len(dailytemp)) * np.arange(len(dailytemp)))

    ind_scenes = [int((d - d0_).days) for d in dates]

    def apply_ia(y, covariance=False):
        exponent = -2 if covariance else -1
        yia = y * np.cos(ia) ** exponent
        return yia
    def extract_predictions(sd, scenes=True):
        if scenes:
            s = sd['s'][:, ind_scenes[1:]]
        else:
            s = sd['s']
        s -= sd['s'][:, ind_scenes[0]][:, np.newaxis]
        return apply_ia(s)

    C = apply_ia(res['C'], covariance=True)[np.newaxis, ...]

    strat = StratigraphyMultiple(StefanStratigraphySmoothingSpline(N=10000), Nbatch=3)
    stefandict = stefan_stratigraphy(dailytemp, strat, **stefanparams)
    s_obs_pred = extract_predictions(stefandict)
    stratsim = StefanStratigraphySmoothingSpline(N=10, seed=114)  #122 #114
    stratsim = StefanStratigraphyConstantE(N=10, seed=31)#29
    stefandictsim = stefan_stratigraphy(dailytemp, stratsim, **stefanparams)
    s_sim_pred = extract_predictions(stefandictsim)

    # run inference
    from inference import psislw, lw_mvnormal, expectation, sumlogs, quantile
    jsim = 0
    # todo: add observation noise
    s_sim_pred_jsim = s_sim_pred[jsim, ...][np.newaxis, ...]
    e_sim_jsim = stefandictsim['e'][jsim, ...]
    lw = lw_mvnormal(s_sim_pred_jsim, C, s_obs_pred)
    lw_ps, _ = psislw(lw)
    e_est = expectation(stefandict['e'], lw_ps, normalize=True)
    s_obs_pred_est = expectation(s_obs_pred, lw_ps, normalize=True)
    s_est = expectation(
        extract_predictions(stefandict, scenes=False), lw_ps, normalize=True)
    yf_est = expectation(stefandict['yf'], lw_ps, normalize=True)

    e_l = quantile(stefandict['e'], lw_ps, 0.1)
    e_h = quantile(stefandict['e'], lw_ps, 0.9)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=2, sharey=False)
    fig.set_size_inches((8, 3), forward=True)
    days = np.arange(len(dailytemp))
    ygrid = np.arange(0, stefandict['depth'], step=stefandict['dy'])
    axs[0].plot(days[ind_scenes[1:]], s_sim_pred_jsim[0, :], lw=0.0, c='k', alpha=0.6,
                marker='o', mfc='k', mec='none', ms=4)
    axs[0].plot(days, s_est[0, :], lw=1.0, c='#999999', alpha=0.6)
#     axs[0].plot(days, s[ind_large[0, -1], :], lw=0.5, c='#ccccff', alpha=0.5)
#     axs[0].plot(days, s[ind_large[0, -2], :], lw=0.5, c='#ccccff', alpha=0.5)
#     axs[0].plot(days, s[ind_large[0, 9000], :], lw=0.5, c='#ffcccc', alpha=0.5)
    axs[1].plot(e_est[0, :], ygrid, lw=1.0, c='#999999', alpha=0.6)
    axs[1].plot(e_sim_jsim, ygrid, lw=1.0, c='#000000', alpha=0.6)
    axs[1].plot(e_l, ygrid, lw=0.5, c='#9999ee', alpha=0.6)
    axs[1].plot(e_h, ygrid, lw=0.5, c='#9999ee', alpha=0.6)

    plt.show()

    # loop over samples
    # extract quantitative comparisons; look at near-surface issues
    #

if __name__ == '__main__':
    simulation()
