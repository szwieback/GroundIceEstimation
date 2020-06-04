import os
import pandas as pd
import numpy as np
import datetime

from scipy.stats import beta
from numpy.random import RandomState

from pathnames import paths
from simulation.stefan import stefan_ens, constants, stefan_integral_balance

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

class SoilParameterEnsemble():
    def __init__(self):
        pass

class StefanParameterEnsemble():
    def __init__(self, rs=None):
        self.depth = 1.5
        self.dy = 2e-3
        self.Ne = 10000
        self.Nb = 15
        self.expb = 2
        self.rs = rs if rs is not None else RandomState(1)
        self.e_params = {'alpha_shape': 1.0, 'beta_shape': 2.0, 'high_scale': 0.8,
                         'alpha_shift': 0.1, 'beta_shift': 2.0}
        self.wsat_params = {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8,
                            'high_below': 1.0}
        # soil only; i.e. without excess ice
        self.soil_params = {'high_horizon': 0.3, 'low_horizon': 0.1, 'organic_above': 0.1,
                            'mineral_above': 0.05, 'mineral_below': 0.3,
                            'organic_below': 0.05}
        # also includes correction factor (Ste > 0, T0 < 0)
        self.n_factor_params = {'high': 0.9, 'low': 0.7, 'alphabeta': 2.0}
        self.constants = constants
        self.stratigraphy = {}

    @property
    def frozen(self):
        # could be turned into ensemble too
        return {'kf': 1.9 * np.ones(self.Ne), 'Cf': 1.5e6 * np.ones(self.Ne), 
                'Tf': -3 * np.ones(self.Ne)}

    def _cpoints(self):
        cpoints = np.linspace(0.08, 1.0, num=self.Nb - 1) ** self.expb * self.depth
        cpoints[0] = 0
        return cpoints

    def _ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

    def _spline_basis(self):
        from scipy.interpolate import BSpline
        cpoints = self._cpoints()
        knots = np.concatenate(([0, 0], cpoints , [self.depth, self.depth]))
        c = np.eye(self.Nb)
        bspline = BSpline(knots, c, k=2, extrapolate=False)
        bsplinead = bspline.antiderivative()
        yg = self._ygrid()
        basis = np.diff(bsplinead(yg), axis=0, prepend=0) / self.dy
        return basis

    def _draw_e(self):
        # draw Bspline coefficients; all independent; 0, 1
        c_shape = self.rs.beta(self.e_params['alpha_shape'], self.e_params['beta_shape'],
                               size=(self.Ne, self.Nb))
        c_scale = self.rs.uniform(high=self.e_params['high_scale'], size=(self.Ne, 1))
        c_shift = self.rs.beta(self.e_params['alpha_shift'], self.e_params['beta_shift'],
                               size=(self.Ne, 1))
        c = c_shift + (1 - c_shift) * c_scale * c_shape
        basis = self._spline_basis()
        e = np.einsum('ij, kj', c, basis)
        return e

    def _draw_organic_depth(self):
        od = self.rs.uniform(low=self.soil_params['low_horizon'],
                             high=self.soil_params['high_horizon'], size=(self.Ne,))
        return od

    def _draw_mow(self, od, e):
        # currently: determinstic given od
        ind_above = self._ygrid()[np.newaxis, :] < od[:, np.newaxis]
        ind_below = np.logical_not(ind_above)
        m = np.zeros_like(ind_above, dtype=np.float64)
        o = np.zeros_like(ind_above, dtype=np.float64)
        sat = np.zeros_like(ind_above, dtype=np.float64)
        np.putmask(m, ind_above, (1 - e) * self.soil_params['mineral_above'])
        np.putmask(o, ind_above, (1 - e) * self.soil_params['organic_above'])
        np.putmask(m, ind_below, (1 - e) * self.soil_params['mineral_below'])
        np.putmask(o, ind_below, (1 - e) * self.soil_params['organic_below'])
        sat_above = self.rs.uniform(low=self.wsat_params['low_above'],
                                    high=self.wsat_params['high_above'], size=(self.Ne,))
        sat_below = self.rs.uniform(low=self.wsat_params['low_below'],
                                    high=self.wsat_params['high_below'], size=(self.Ne,))
        print(sat_below.shape)
        np.putmask(sat, ind_above, sat_above[:, np.newaxis] * np.ones_like(sat))
        np.putmask(sat, ind_below, sat_below[:, np.newaxis] * np.ones_like(sat))
        w = (1 - e - m - o) * sat
        return m, o, w

    def _draw_n_factor(self):
        beta = self.rs.beta(self.n_factor_params['alphabeta'],
                            self.n_factor_params['alphabeta'], size=(self.Ne,))
        n_factor = (self.n_factor_params['low']
                    +beta * (self.n_factor_params['high'] - self.n_factor_params['low']))
        return n_factor

    def _thermal_conductivity_thawed(self):
        if any([con not in self.stratigraphy for con in ['m', 'o', 'w']]):
            raise AttributeError('Stratigraphy needs to be assigned first')
        # Cosenza; neglect air
        depthf = (1 - self.stratigraphy['e'])
        # actually each constituent should be divided by depthf
        k = (((self.constants['km']) ** 0.5 * self.stratigraphy['m'] +
             (self.constants['ko']) ** 0.5 * self.stratigraphy['o'] +
             (self.constants['kw']) ** 0.5 * self.stratigraphy['w']
             ) ** 2) / depthf
        ikg = k ** (-1)
        ik = np.cumsum(ikg * depthf, axis=1) / np.cumsum(depthf, axis=1)
        k0 = k[:, 1]
        k0ik = k0[:, np.newaxis] * ik
        dict_k = {'k0ik': k0ik, 'k0': k0}
        return dict_k

    def draw_stratigraphy(self):
        e = spe._draw_e()
        od = spe._draw_organic_depth()
        m, o, w = spe._draw_mow(od, e)
        n_factor = spe._draw_n_factor()
        self.stratigraphy = {'e': e, 'm': m, 'o': o, 'w': w, 'od': od,
                             'n_factor': n_factor}
        self.stratigraphy.update(self._thermal_conductivity_thawed())

    @property
    def params(self):
        return {**self.stratigraphy, **self.constants, 'depth': self.depth, 'dy': self.dy,
                **self.frozen}

if __name__ == '__main__':
    df = load_forcing()
    d0 = '2019-05-15'
    d1 = '2019-09-15'
    dailytemp = (df.resample('D').mean())['air_temp_5m'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    s_, yf_ = stefantest(dailytemp)

    spe = StefanParameterEnsemble()
    spe.draw_stratigraphy()
    dailytemp_ens = np.zeros((spe.Ne, len(dailytemp)))
    dailytemp_ens[:, :] = np.array(dailytemp)[np.newaxis, :]



    from timeit import timeit
    fun_wrapped = lambda: stefan_ens(dailytemp_ens, params=spe.params)
    print(f'{timeit(fun_wrapped, number=1)}')
    s, yf, _ = stefan_ens(dailytemp_ens, params=spe.params)
#     import matplotlib.pyplot as plt
#     plt.hist(s[:, -1])
#     plt.show()
    print(np.percentile(yf[:, -1], (5, 50, 95)))
#     print(s[0, :] - s_)
    fun_wrapped = lambda: stefan_integral_balance(dailytemp_ens, params=spe.params)
    print(f'{timeit(fun_wrapped, number=1)}')
    # TODO: k,
    s, yf = stefan_integral_balance(dailytemp_ens, params=spe.params)


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
    e_est = np.einsum('ij,jk->ik', np.exp(lw_ps), spe.params['e'])
    s_obstime_est = np.einsum('ij,jk->ik', np.exp(lw_ps), s_obstime)
    print(s_obstime_est)
    print(e_est[:, ::25])

#     print(spe.params['e'][ind_true, ::25])

#     print(e_est)
    # need more variable ice profiles

