'''
Created on Jun 4, 2020

@author: simon
'''
from numpy.random import RandomState
import numpy as np
import warnings

constants_default = {
    'Lvw': 3.34e8, 'km': 3.80, 'ko': 0.25, 'ki': 2.20, 'kw': 0.57, 'ka': 0.024}
params_default_ancillary = {'kf': 1.9, 'Cf': 1.5e6, 'Tf':-4.0, 'Ct': 3.0e6}
params_default_grid = {'dy': 2e-3, 'depth': 1.5}
params_default = {**constants_default, **params_default_ancillary, **params_default_grid}
params_default_distribution = {
    'Nb': 15, 'expb': 2.0, 'b0': 0.1, 'bm': 0.7,
    'e': {'alpha_shape': 1.0, 'beta_shape': 2.0, 'high_scale': 0.8, 'alpha_shift': 0.1,
          'beta_shift': 2.0},
    'wsat': {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.3, 'low_horizon': 0.1, 'organic_above': 0.1,
             'mineral_above': 0.05, 'mineral_below': 0.3, 'organic_below': 0.05},
    'n_factor': {'high': 0.95, 'low': 0.8, 'alphabeta': 2.0}}
params_default_distribution = {
    'Nb': 12, 'expb': 2.0, 'b0': 0.10, 'bm': 0.80,
    'e': {'low': 0.00, 'high': 0.95, 'coeff_mean':3, 'coeff_std': 3, 'coeff_corr': 0.7},
    'wsat': {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
    'soil': {'high_horizon': 0.3, 'low_horizon': 0.1, 'organic_above': 0.1,
             'mineral_above': 0.05, 'mineral_below': 0.3, 'organic_below': 0.05},
    'n_factor': {'high': 0.95, 'low': 0.85, 'alphabeta': 2.0}}

class Stratigraphy():
    def __init__(
            self, dy=None, depth=None, N=10000, dist=None, ancillary=None, rs=None, seed=1,
            constants=None):
        pass
    def _update_rs(self, rs):
        self.rs = rs

class StratigraphyMultiple():
    def __init__(self, strat, Ns=100, seed0=1):
        self.strat = strat
        self.Ns = Ns
        self.seed0 = seed0
        
    def params(self, fields=None):
        params0 = {**self.strat.constants, 'depth': self.strat.depth, 'dy': self.strat.dy}
        params_strat = {}
        for js in range(self.Ns):
            self.strat._update_rs(RandomState(seed=self.seed0+js))
            self.strat.draw_stratigraphy(verbose=False)
            params_stratjs = self.strat.stratigraphy
            for field in params_stratjs:
                if fields is None or field in fields:
                    if js == 0 or np.isscalar(params_stratjs[field]):
                        params_strat[field] = params_stratjs[field]
                    else:
                        params_strat[field] = np.concatenate(
                            (params_strat[field], params_stratjs[field]), axis=0)
        return {**params0, **params_strat}

class StefanStratigraphy(Stratigraphy):
    def __init__(
            self, dy=None, depth=None, N=10000, dist=None, ancillary=None, rs=None, seed=1,
            constants=None):
        self.depth = depth if depth is not None else params_default_grid['depth']
        self.dy = dy if dy is not None else params_default_grid['dy']
        self.rs = rs if rs is not None else RandomState(seed=seed)
        dist_ = dist if dist is not None else params_default_distribution
        self.N = N
        self.Nb = dist_['Nb']
        self.b0 = dist_['b0']
        self.bm = dist_['bm']
        self.expb = dist_['expb']
        self.e_params = dist_['e']
        self.wsat_params = dist_['wsat']
        # soil only; i.e. without excess ice
        self.soil_params = dist_['soil']
        # can also account for correction factor (Ste > 0, T0 < 0) if basic stefan is used
        self.n_factor_params = dist_['n_factor']
        self.ancillary = (
            ancillary if ancillary is not None else params_default_ancillary)
        self.constants = constants if constants is not None else constants_default
        self.stratigraphy = {}
        self.dtype = np.float32

    def _cpoints(self):
        cpoints = np.linspace(self.b0, self.bm, num=self.Nb - 2) ** self.expb
        cpoints[0] = 0
        cpoints = np.concatenate((cpoints, [1])) * self.depth
        return cpoints

    @property
    def _ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

    def cell_index(self, depth):
        d_ = np.atleast_1d(depth)
        ind = np.zeros(d_.shape, dtype=np.uint32) - 1
        yg_ = self._ygrid[(np.newaxis,) * len(d_.shape) + (slice(None),)]
        ind_ = np.argmin(np.abs(d_[..., np.newaxis] - yg_), axis=-1)
        valid = np.logical_and(d_ >= 0, d_ <= self.depth)
        ind[valid] = ind_[valid]
        if not isinstance(depth, (list, tuple, np.ndarray)):
            ind = ind[0]
        return ind

    def _spline_basis(self):
        from scipy.interpolate import BSpline
        cpoints = self._cpoints()
        knots = np.concatenate(([0, 0], cpoints , [self.depth, self.depth]))
        c = np.eye(self.Nb)
        bspline = BSpline(knots, c, k=2, extrapolate=False)
        # integrate than difference
        # focus on average rather than at grid point
        # partition of unity is preserved (except 0 element)
        bsplinead = bspline.antiderivative()
        yg = self._ygrid
        basis = np.diff(bsplinead(yg), axis=0, prepend=0) / self.dy
        return basis

    def _draw_e(self):
        # draw Bspline coefficients; all independent; 0, 1
        c_shape = self.rs.beta(self.e_params['alpha_shape'], self.e_params['beta_shape'],
                               size=(self.N, self.Nb))
        c_scale = self.rs.uniform(high=self.e_params['high_scale'], size=(self.N, 1))
        c_shift = self.rs.beta(self.e_params['alpha_shift'], self.e_params['beta_shift'],
                               size=(self.N, 1))
        c_shift = c_shift * 0
        import warnings
        warnings.warn('c_shift set to 0')
        c = c_shift + (1 - c_shift) * c_scale * c_shape
        basis = self._spline_basis()
        e = np.einsum('ij, kj', c, basis)
        return e

    def _draw_organic_depth(self):
        od = self.rs.uniform(low=self.soil_params['low_horizon'],
                             high=self.soil_params['high_horizon'], size=(self.N,))
        return od

    def _draw_mow(self, od, e):
        # currently: determinstic given od
        ind_above = self._ygrid[np.newaxis, :] < od[:, np.newaxis]
        ind_below = np.logical_not(ind_above)
        m = np.zeros_like(ind_above, dtype=np.float64)
        o = np.zeros_like(ind_above, dtype=np.float64)
        sat = np.zeros_like(ind_above, dtype=np.float64)
        np.putmask(m, ind_above, (1 - e) * self.soil_params['mineral_above'])
        np.putmask(o, ind_above, (1 - e) * self.soil_params['organic_above'])
        np.putmask(m, ind_below, (1 - e) * self.soil_params['mineral_below'])
        np.putmask(o, ind_below, (1 - e) * self.soil_params['organic_below'])
        sat_above = self.rs.uniform(low=self.wsat_params['low_above'],
                                    high=self.wsat_params['high_above'], size=(self.N,))
        sat_below = self.rs.uniform(low=self.wsat_params['low_below'],
                                    high=self.wsat_params['high_below'], size=(self.N,))
        np.putmask(sat, ind_above, sat_above[:, np.newaxis] * np.ones_like(sat))
        np.putmask(sat, ind_below, sat_below[:, np.newaxis] * np.ones_like(sat))
        w = (1 - e - m - o) * sat
        return m, o, w

    def _draw_n_factor(self):
        beta = self.rs.beta(self.n_factor_params['alphabeta'],
                            self.n_factor_params['alphabeta'], size=(self.N,))
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
        dict_k = {'k0ik': k0ik.astype(self.dtype), 'k0': k0.astype(self.dtype)}
        return dict_k

    def _ancillary(self):
        # uniform for now
        p = {'kf': self.ancillary['kf'],
             'Cf': self.ancillary['Cf'],
             'Tf': self.ancillary['Tf'],
             'Ct': self.ancillary['Ct']}
        return p

    def draw_stratigraphy(self, verbose=True):
        if len(self.stratigraphy) > 0 and verbose:
            warnings.warn("Overwriting stratigraphy", RuntimeWarning)
        e = self._draw_e().astype(self.dtype)
        od = self._draw_organic_depth().astype(self.dtype)
        m, o, w = self._draw_mow(od, e)
        n_factor = self._draw_n_factor().astype(self.dtype)
        p = self._ancillary()
        self.stratigraphy = {
            'e': e, 'm': m.astype(self.dtype), 'o': o.astype(self.dtype),
            'w': w.astype(self.dtype), 'od': od, 'n_factor': n_factor, **p}
        self.stratigraphy.update(self._thermal_conductivity_thawed())

    def params(self):
        return {**self.constants, **self.stratigraphy, 'depth': self.depth, 'dy': self.dy}

class StefanStratigraphySmoothingSpline(StefanStratigraphy):
    def _draw_e(self):
        # draw Bspline coefficients; all independent; 0, 1
        ehigh, elow = self.e_params['low'], self.e_params['high']
        coeffmean, coeffstd = self.e_params['coeff_mean'], self.e_params['coeff_std']
        coeffcorr = self.e_params['coeff_corr']
        coeff = self.rs.normal(size=(self.N, self.Nb))
        for jcoeff in range(1, self.Nb):
            coeff[:, jcoeff] += coeffcorr * coeff[:, jcoeff - 1]
        coeff *= (1 - coeffcorr ** 2) ** (0.5)  # make asymptotic variance 1
        coeff = coeffmean + coeff * coeffstd
        basis = self._spline_basis()
        elogit = np.einsum('ij, kj', coeff, basis)
        e = (1 + np.exp(-elogit)) ** (-1) * (ehigh - elow) + elow
        return e

if __name__ == '__main__':
    strat = StefanStratigraphySmoothingSpline()
    print(strat._cpoints())
#     print(np.sum(strat._spline_basis(), axis=1))
    e = strat._draw_e()
    import matplotlib.pyplot as plt
    ygrid = strat._ygrid
    plt.plot(ygrid, e[0:50, :].T, alpha=0.5)
    plt.show()

