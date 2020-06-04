import os
import pandas as pd
import numpy as np
import datetime

from pathnames import paths
from simulation.stefan import stefan_ens, stefan_integral_balance
from simulation.stratigraphy import StefanStratigraphy

def load_forcing(year=2019):
    assert year == 2019
    fn = os.path.join(paths['forcing'], 'toolik2019', '1-hour_data.csv')
    def dp(d, t):
        assert isinstance(t, str)
        t_ = str(int(t) - 100)
        dt = d + ' ' + t_.zfill(4)
        dtp = datetime.datetime.strptime(dt, '%Y-%m-%d %H%M')
        dtp = dtp + pd.Timedelta(1, 'h')
        return dtp

    df = pd.read_csv(fn, parse_dates={'datetime': ['date', 'hour']}, date_parser=dp)
    df = df.set_index('datetime')
    return df

def stefantest(dailytemp):
    k0 = 0.4
    ik = lambda yg: np.ones_like(yg)
    depth = 10.0
    dy = 0.0025
    L_v = 3.34e8
    yg = np.arange(0, depth, step=dy)
    wg = 0.4 * np.ones_like(yg)
    eg = 0.1 * np.ones_like(yg)
    ikg = ik(yg)
    Lg = L_v * (wg + eg)
    sg = np.cumsum(eg * dy)
    upsg = yg - sg
    k0d = k0 * 3600 * 24
    fac = 1e-9
    n_factor = 0.9

    tterm = np.cumsum((k0d * fac * n_factor) * np.array(dailytemp))
    yterm = np.cumsum(ikg * (Lg * fac) * upsg * dy)
    yf = np.ones_like(tterm) * -1
    s = np.zeros_like(tterm)
    tind = 0
    yind = 0
    while tind < tterm.shape[0]:
        if tterm[tind] >= yterm[yind]:
            yind = yind + 1
            if yind == yterm.shape[0]:
                raise ValueError('depth not big enough')
        else:
            yf[tind] = yind * dy
            s[tind] = sg[yind]
            tind = tind + 1
    return s, yf

if __name__ == '__main__':
    df = load_forcing()
    d0 = '2019-05-15'
    d1 = '2019-09-15'
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    s_, yf_ = stefantest(dailytemp)

    strat = StefanStratigraphy()
    strat.draw_stratigraphy()
    dailytemp_ens = np.zeros((strat.Ne, len(dailytemp)))
    dailytemp_ens[:, :] = np.array(dailytemp)[np.newaxis, :]

    from timeit import timeit
    fun_wrapped = lambda: stefan_ens(dailytemp_ens, params=strat.params)
    print(f'{timeit(fun_wrapped, number=1)}')
    s, yf, _ = stefan_ens(dailytemp_ens, params=strat.params)
#     import matplotlib.pyplot as plt
#     plt.hist(s[:, -1])
#     plt.show()
    print(np.percentile(yf[:, -1], (5, 50, 95)))
#     print(s[0, :] - s_)
    fun_wrapped = lambda: stefan_integral_balance(dailytemp_ens, params=strat.params)
    print(f'{timeit(fun_wrapped, number=1)}')
    # TODO: k,
    s, yf = stefan_integral_balance(dailytemp_ens, params=strat.params)


    unc = 0.01
    ind_true = 0
    s_obstime = s[:, 16::11]
    rs = np.random.RandomState(seed=123)
    s_obs = s_obstime[ind_true, :]
    s_obs = np.array([0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
    s_obs = s_obs + unc * rs.normal(size=(1, s_obstime.shape[1]))
    print(s_obs)
    from inference import psislw, lw_mvnormal
    lw = lw_mvnormal(s_obs, unc**2*np.eye(s_obstime.shape[1])[np.newaxis, ...], s_obstime)
    lw_ps, _ = psislw(lw)
    e_est = np.einsum('ij,jk->ik', np.exp(lw_ps), strat.params['e'])
    s_obstime_est = np.einsum('ij,jk->ik', np.exp(lw_ps), s_obstime)
    print(s_obstime_est)
    print(e_est[:, ::25])

#     print(spe.params['e'][ind_true, ::25])

#     print(e_est)
    # need more variable ice profiles

