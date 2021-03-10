'''
Created on Jul 15, 2020

@author: simon
'''
import pandas as pd
import numpy as np
from simulation.toolik import load_forcing
from simulation.stefan import stefan_ens, stefan_integral_balance
from simulation.stefan_single import stefan_integral_balance_single
from simulation.stratigraphy import (StefanStratigraphy, StefanStratigraphySmoothingSpline,
                                     StratigraphyMultiple)

fieldsdef = ('e','depth', 'dy')

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
                

if __name__ == '__main__':
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
    