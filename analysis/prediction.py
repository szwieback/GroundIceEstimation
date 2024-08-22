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
    def __init__(self, strat, predictor, geom=None, results=None):
        self.strat = strat
        self.predictor = predictor
        self.geom = geom
        self.results = results

    @property
    def N(self):
        return self.strat.N

    def predict(self, forcing, n_jobs=-8, **kwargs):
        strat = self.strat
        self.results = self._predict(strat, forcing, n_jobs=n_jobs, **kwargs)

    def _predict(self, strat, forcing, n_jobs=-8, **kwargs):
        results = {}
        if strat.Nbatch == 0:
            results = self.predictor.predict(
                forcing, strat.params(), geom=self.geom, **kwargs)
        else:
            from joblib import Parallel, delayed
            def _res(batch):
                res = self.predictor.predict(
                    forcing, strat.params(batch=batch), geom=self.geom, **kwargs)
                return res
            rl = Parallel(n_jobs=n_jobs)(delayed(_res)(b) for b in range(strat.Nbatch))
            for res_batch in rl:
                for k in res_batch:
                    if k in results:
                        if not np.isscalar(self.results[k]):
                            results[k] = np.concatenate((results[k], res_batch[k]), axis=0)
                    else:
                        results[k] = res_batch[k]
        return results

    def predict_mean_period(self, indranges, param='e'):
        if isinstance(param, str):
            self._predict_mean_period(indranges, param=param)            
        else:
            for _p in param: self.predict_mean_period(indranges, _p)

    def _predict_mean_period(self, indranges, param='e'):
        mp = self._mean_period(self.results, indranges, param=param)
        self.results[f'{param}_mean_period'] = mp

    def _mean_period(self, results, indranges, param='e'):
        ygrid = self.ygrid
        yf = results['yf']
        p = results[param]
        p_mean = []
        for indrange in indranges:
            yfrange = yf[:, indrange]
            invalid = np.logical_or(
                yfrange[:, 0][:, np.newaxis] > ygrid[np.newaxis,:],
                yfrange[:, 1][:, np.newaxis] < ygrid[np.newaxis,:])
            p_ = p.copy()
            np.putmask(p_, invalid, np.nan)
            p_mean.append(np.nanmean(p_, axis=1))
        p_mean = np.stack(p_mean, axis=-1)
        return p_mean

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
            self, indices, field='s_los', C_obs=None, rng=None, reference_only=False, **kwargs):
        # need kwargs for downstream compatibility
        pred = self._extract_predictions(
            self.results, indices, field=field, C_obs=C_obs, rng=rng, reference_only=reference_only)
        return pred

    @classmethod
    def _extract_predictions(
            cls, results, indices, field='s_los', C_obs=None, rng=None, reference_only=False):
        # nn interpolation from time steps to observation epochs
        # indices: ind of time steps
        # C not None: add measurement noise
        if not reference_only:
            s = results[field][:, indices[1:]]
        else:
            s = results[field]
        s -= results[field][:, indices[0]][:, np.newaxis]
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

class MulticlassPredictionEnsemble(PredictionEnsemble):
    def __init__(self, strats, predictor, geom=None, results=None):
        # strats is dictionary of stratigraphies
        self._check_strats(strats)
        self.strats = strats
        self.predictor = predictor
        self.geom = geom
        self.results = results

    @property
    def classnames(self):
        return tuple(self.strats.keys())
    
    def __getitem__(self, cn):
        if cn not in self.classnames: raise ValueError(f'Class name {cn} not found')
        results = None if self.results is None else self.results[cn]
        pe = PredictionEnsemble(self.strats[cn], self.predictor, self.geom, results=results)
        return pe

    @classmethod
    def _check_strats(cls, strats):
        s0 = strats[tuple(strats.keys())[0]]
        for s in strats.values():
            assert s0.depth == s.depth
            assert s0.dy == s.dy
            assert s0.N == s.N
            
    @property
    def depth(self):
        return self.strats[self.classnames[0]].depth

    @property
    def dy(self):
        return self.strats[self.classnames[0]].dy

    @property
    def N(self):
        return self.strats[self.classnames[0]].N

    def _predict_mean_period(self, indranges, param='e'):
        for sc in self.strats:
            results = self.results[sc]
            mp = self._mean_period(results, indranges, param=param)        
            self.results[sc][f'{param}_mean_period'] = mp

    def predict(self, forcing, n_jobs=-8, **kwargs):
        self.results = {}
        for sc in self.strats:
            self.results[sc] = self._predict(self.strats[sc], forcing, n_jobs=n_jobs, **kwargs)

    def extract_predictions(
            self, indices, field='s_los', C_obs=None, rng=None, reference_only=False, **kwargs):
        try:
            ec = kwargs['ec']
        except:
            raise ValueError('ec needs to be provided for MulticlassPredictionEnsemble')
        results = self.results[ec]
        pred = self._extract_predictions(
            results, indices, field=field, C_obs=C_obs, rng=rng, reference_only=reference_only)
        return pred

