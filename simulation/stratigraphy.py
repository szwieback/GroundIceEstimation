'''
Created on Jun 4, 2020

@author: simon
'''
from numpy.random import RandomState
import numpy as np
import warnings

constants = {'Lvw': 3.34e8, 'km': 3.80, 'ko': 0.25, 'ki': 2.20, 'kw': 0.57, 'ka': 0.024}
params_default_frozen = {'kf': 1.9, 'Cf': 1.5e6, 'Tf':-4.0}
params_default_grid = {'dy': 2e-3, 'depth': 2}
params_default = {**constants, **params_default_frozen, **params_default_grid}

class Stratigraphy():
    def __init__(self):
        pass

class StefanStratigraphy(Stratigraphy):
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
                'Tf':-3 * np.ones(self.Ne)}

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
        if len(self.stratigraphy) > 0:
            warnings.warn("Overwriting stratigraphy", RuntimeWarning)
        e = self._draw_e()
        od = self._draw_organic_depth()
        m, o, w = self._draw_mow(od, e)
        n_factor = self._draw_n_factor()
        self.stratigraphy = {'e': e, 'm': m, 'o': o, 'w': w, 'od': od,
                             'n_factor': n_factor}
        self.stratigraphy.update(self._thermal_conductivity_thawed())

    @property
    def params(self):
        return {**self.stratigraphy, **self.constants, 'depth': self.depth, 'dy': self.dy,
                **self.frozen}
