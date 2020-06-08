import os
import pandas as pd
import numpy as np
import datetime
from copy import deepcopy

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
    strat_ = StefanStratigraphy(N=1)
    strat_.draw_stratigraphy()
    e = strat_.stratigraphy['e']
#     e[:, 0:80] = 0.2
#     e[:, 80:160] = 0.01
#     e[:, 160:] = 0.5
    e[:, 0:60] = 0.2
    e[:, 60:180] = 0.0
    e[:, 180:] = 0.1
    strat_.stratigraphy['e'] = e
    dailytemp_ens = np.zeros((strat.N, len(dailytemp)))
    dailytemp_ens[:, :] = np.array(dailytemp)[np.newaxis, :]

    from timeit import timeit
    fun_wrapped = lambda: stefan_ens(dailytemp_ens, params=strat.params)
#     print(f'{timeit(fun_wrapped, number=1)}')
#     s, yf, _ = stefan_ens(dailytemp_ens, params=strat.params)
#     import matplotlib.pyplot as plt
#     plt.hist(s[:, -1])
#     plt.show()
#     print(s[0, :] - s_)
    fun_wrapped = lambda: stefan_integral_balance(dailytemp_ens, params=strat.params)
#     print(f'{timeit(fun_wrapped, number=1)}')
    s, yf = stefan_integral_balance(dailytemp_ens, params=strat.params, steps=1)
    s_, yf_ = stefan_integral_balance(dailytemp_ens[0:1, :], params=strat_.params, steps=1)
    
    unc = 0.003
    ind_true = -1#2, 4, 8
    ind_obs = np.arange(s.shape[1])[16::11]
    s_obstime = s[:, ind_obs]
    s__obstime = s_[:, ind_obs]
    rs = np.random.RandomState(seed=123)
    s_obs = s__obstime[0, :]
#     s_obs = np.array([0.03, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])
#     s_obs = np.array([0.01, 0.03, 0.05, 0.06, 0.06, 0.06, 0.08, 0.10, 0.11, 0.12]) / 2
    s_obs = s_obs + unc * rs.normal(size=(1, s_obstime.shape[1]))
    print(s_obs)
    from inference import psislw, lw_mvnormal, expectation
    lw = lw_mvnormal(s_obs, unc ** 2 * np.eye(s_obstime.shape[1])[np.newaxis, ...], s_obstime)
    lw_ps, _ = psislw(lw)
    e_est = expectation(strat.params['e'], lw_ps, normalize=True)
    s_obstime_est = expectation(s_obstime, lw_ps, normalize=True)
    s_est = expectation(s, lw_ps, normalize=True)

    yf_est = expectation(yf, lw_ps, normalize=True)
    print(yf_est[:, -1])
    ind_large = np.argsort(lw_ps, axis=1)
#     print(e_est[:, :strat.cell_index(yf_est[0,-1]):25])

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=2, sharey=False)
    fig.set_size_inches((8,3), forward=True)
    days = np.arange(len(dailytemp))
    axs[0].plot(days[ind_obs], s_obs[0, :], lw=0.0, c='k', alpha=0.6, marker='o',
            mfc='k', mec='none', ms=4)
    axs[0].plot(days, s_est[0, :], lw=1.0, c='#999999', alpha=0.6)
#     axs[0].plot(days, s[ind_large[0, -1], :], lw=0.5, c='#ccccff', alpha=0.5)
#     axs[0].plot(days, s[ind_large[0, -2], :], lw=0.5, c='#ccccff', alpha=0.5)
#     axs[0].plot(days, s[ind_large[0, 9000], :], lw=0.5, c='#ffcccc', alpha=0.5)
    axs[0].plot(days, s_[0, :], lw=0.6, c='#ccffcc', alpha=1.0)
    axs[1].plot(e_est[0, :], strat._ygrid, lw=1.0, c='#999999', alpha=0.6)
    axs[1].plot(e[0, :], strat._ygrid, lw=0.6, c='#ccffcc', alpha=1.0)
    plt.show()

#     print(spe.params['e'][ind_true, ::25])

#     print(e_est)
    # need more variable ice profiles

