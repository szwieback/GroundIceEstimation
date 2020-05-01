import os
import pandas as pd
import numpy as np

from pathnames import paths

def load_forcing(year=2019):
    assert year == 2019
    fn = os.path.join(paths['forcing'], 'toolik2019', '1-hour_data.csv')
    def dp(d, t):
        assert isinstance(t, str)
        t_ = str(int(t) - 100)
        dt = d + ' ' + t_.zfill(4)
        dtp = pd.datetime.strptime(dt, '%Y-%m-%d %H%M')
        dtp = dtp + pd.Timedelta(1, 'h')
        return dtp

    df = pd.read_csv(fn, parse_dates={'datetime': ['date', 'hour']}, date_parser=dp)
    df = df.set_index('datetime')
    return df

def stefantest(dailytemp):
    k0 = 0.4
    ik = lambda yg: np.ones_like(yg)
    depth = 5.0
    dy = 0.001
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

    tterm = np.cumsum((k0d * fac) * np.array(dailytemp))
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
    
    Ne = 100 
    dailytemp_ens = np.zeros((Ne, len(dailytemp)))
    dailytemp_ens[:, :] = np.array(dailytemp)[np.newaxis, :]
    
    from simulation.stefan import stefan, stefan_ens
    from timeit import timeit
    fun_wrapped = lambda: stefan_ens(dailytemp_ens)
    print(f'{timeit(fun_wrapped, number=1)}')
    s, yf = stefan_ens(dailytemp_ens)
    print(s[0, :] - s_)
#     s, yf = stefan(dailytemp)

