'''
Created on Nov 5, 2021

@author: simon
'''
from simulation import stefan_integral_balance

import numpy as np

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
        self.strat = strat
        self.predictor = predictor
        self.geom = geom
        self.results = None

    def predict(self, forcing, n_jobs=-8, **kwargs):
        self.results = {}
        if self.strat.Nbatch == 0:
            self.results = self.predictor.predict(
                forcing, self.strat.params(), geom=self.geom, **kwargs)
        else:
            from joblib import Parallel, delayed
            def _res(batch):
                res = self.predictor.predict(
                    forcing, self.strat.params(batch=batch), geom=self.geom, **kwargs)
                return res
            rl = Parallel(n_jobs=n_jobs)(delayed(_res)(b) for b in range(self.strat.Nbatch))
            for res_batch in rl:
                for k in res_batch:
                    if k in self.results:
                        if not np.isscalar(self.results[k]):
                            self.results[k] = np.concatenate(
                                (self.results[k], res_batch[k]), axis=0)
                    else:
                        self.results[k] = res_batch[k]

    @property
    def depth(self):
        return self.results['depth']

    @property
    def dy(self):
        return self.results['dy']

    @property
    def ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

    def extract_predictions(
            self, indices, field='s_los', C_obs=None, rng=None, reference_only=False):
        # nn interpolation from time steps to observation epochs
        # indices: ind of time steps
        # C not None: add measurement noise
        if not reference_only:
            s = self.results[field][:, indices[1:]]
        else:
            s = self.results[field]
        s -= self.results[field][:, indices[0]][:, np.newaxis]
        if C_obs is not None:
            if rng is None:
                rng = np.random.default_rng(seed=1)
            try:
                assert C_obs.shape[0] == s.shape[1]
                obs_noise = rng.multivariate_normal(
                    np.zeros(s.shape[1]), C_obs, size=(s.shape[0],))
                s += obs_noise
            except:
                s += np.nan
        return s