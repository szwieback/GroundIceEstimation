'''
Created on Jul 15, 2020

@author: simon
'''
import pandas as pd
import numpy as np
from simulation.toolik import load_forcing
from simulation.stefan import stefan_ens, stefan_integral_balance
from simulation.stefan_single import stefan_integral_balance_single
from simulation.stratigraphy import StefanStratigraphy, StefanStratigraphySmoothingSpline

if __name__ == '__main__':
    df = load_forcing()
    d0 = '2019-05-15'
    d1 = '2019-09-15'
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0

    strat = StefanStratigraphySmoothingSpline(N=10000)
    strat.draw_stratigraphy()
    
#     from timeit import timeit
#     fun_wrapped = lambda: stefan_integral_balance(
#         dailytemp, params=strat.params, steps=1)
#     print(f'{timeit(fun_wrapped, number=1)}')
#     s, yf = stefan_integral_balance(dailytemp, params=strat.params, steps=0)
    s2, yf2 = stefan_integral_balance(dailytemp, params=strat.params, steps=1)
    print(np.percentile(yf2[:, -1], [10, 50, 90]))
#     print(np.percentile((yf - yf2)[:, -15], [10, 50, 90]))
    
#     effect of C is very small; affects timing slightly

    # stefan: stack option