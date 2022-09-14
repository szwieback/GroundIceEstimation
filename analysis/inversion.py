'''
Created on Sep 14, 2022

@author: simon
'''

from analysis import enforce_directory

import numpy as np
import os

class InversionProcessor():
    # hard-coded Gaussian PSIS
    # uses the same ensemble but different C_obs
    def __init__(self, predens=None, geospatial=None, batch_size=1000):
        self.predens = predens
        self.geospatial = geospatial
        self.batch_size = batch_size

    def _simulated_observations_single(self, ind_scenes, _C_obs):
        s_pred = self.predens.extract_predictions(
            ind_scenes, C_obs=_C_obs, rng=None)  # hardcoded seed for now
        return s_pred

    def _logweights_single(self, ind_scenes, _s_obs, _C_obs, normalize=False):
        from inference import lw_mvnormal, psislw, _normalize
        try:
            s_pred = self._simulated_observations_single(ind_scenes, _C_obs)
            lw = lw_mvnormal(
                _s_obs[np.newaxis, :], _C_obs[np.newaxis, ...], s_pred)
            lw_ps, _ = psislw(lw)
            lw_ps = _normalize(lw_ps, normalize=normalize)
        except:
            lw_ps = np.full((1, s_pred.shape[0]), np.nan)
        return lw_ps

    def _filename(self, path0, ftype, number=None, ext='npy'):
        if path0 is None:
            return None
        else:
            _fn = ftype if number is None else f'{ftype}_{number}'
            return os.path.join(path0, _fn + f'.{ext}')

    def _overwrite(self, fn, overwrite=False):
        try:
            ow = not os.path.exists(fn) or overwrite
        except:
            ow = True
        return ow

    def _logweights_batch(
            self, nbatch, ind_scenes, s_obs_flat, C_obs_flat, normalize=False, pathout=None,
            overwrite=False):
        _fn = self._filename(pathout, 'lw', nbatch)
        if self._overwrite(_fn, overwrite=overwrite):
            n0 = nbatch * self.batch_size
            n1 = min(((nbatch + 1) * self.batch_size, s_obs_flat.shape[-1]))
            _s_obs_batch = s_obs_flat[..., n0:n1].copy()
            _C_obs_batch = C_obs_flat[..., n0:n1].copy()
            lw = []
            for ndiff in range(0, n1 - n0):
                _s, _C = _s_obs_batch[..., ndiff], _C_obs_batch[..., ndiff]
                lw.append(self._logweights_single(ind_scenes, _s, _C, normalize=normalize))
            lw = np.concatenate(lw, axis=0)
            if _fn is not None:
                enforce_directory(_fn)
                np.save(_fn, lw)
        else:
            lw = np.load(_fn)
        return lw

    def logweights(
            self, ind_scenes, s_obs, C_obs, n_jobs=8, normalize=False, pathout=None,
            overwrite=False):
        from joblib import Parallel, delayed
        s_obs_flat = np.reshape(s_obs, (s_obs.shape[0], -1))
        C_obs_flat = np.reshape(C_obs, (C_obs.shape[0], C_obs.shape[1], -1))
        N = s_obs_flat.shape[-1]
        assert C_obs_flat.shape[-1] == N
        Nbatch = np.int64(np.ceil(N / self.batch_size))
        def _res(nbatch):
            return self._logweights_batch(
                nbatch, ind_scenes, s_obs_flat, C_obs_flat, normalize=normalize,
                pathout=pathout, overwrite=overwrite)
        lw = Parallel(n_jobs=n_jobs)(delayed(_res)(nbatch) for nbatch in range(Nbatch))
        lw = np.concatenate(lw, axis=0)
        return lw

    def results(
            self, ind_scenes, s_obs, C_obs, n_jobs=8, normalize=False, pathout=None,
            overwrite=False):
        lw = self.logweights(
            ind_scenes, s_obs, C_obs, n_jobs=n_jobs, normalize=normalize, pathout=pathout,
            overwrite=overwrite)
        lw = np.reshape(lw, s_obs.shape[1:] + (lw.shape[-1],))
        return InversionResults(self.predens, lw)

    @property
    def depth(self):
        return self.predens.depth

    @property
    def dy(self):
        return self.predens.dy

    @property
    def ygrid(self):
        return self.predens.ygrid

class InversionResults():
    def __init__(self, predens, lw, geospatial=None):
        self.predens = predens
        self.lw = lw
        self.geospatial = geospatial

    @property
    def depth(self):
        return self.predens.results['depth']

    @property
    def dy(self):
        return self.predens.results['dy']

    @property
    def ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

    def predictions(self, param='e', p=None):
        if p is None:
            p = self.predens.results[param]
        return p

    def moment(self, param='e', power=1, p=None, normalize=True):
        from inference import expectation
        p = self.predictions(param=param, p=p)
        postmean = expectation(
            np.power(p, power), self.lw, normalize=normalize)
        return postmean

    def variance(self, param='e', p=None, normalize=True):
        # improvement needed to deal with numerical issues
        p = self.predictions(param=param, p=p)
        var = (self.moment(param=None, p=p, power=2, normalize=normalize)
               -self.moment(param=None, p=p, power=1, normalize=normalize) ** 2)
        return var

    def quantile(
            self, quantiles, param='e', smooth=None, steps=8, p=None, method='bisection'):
        from inference import quantile as quant
        p = self.predictions(param=param, p=p)
        postquant = quant(
            p, self.lw, quantiles, method=method, steps=steps, normalize=True,
            smooth=smooth)
        return postquant

    def frac_thawed(self, ind_scene):
        from inference import _normalize
        yf = self.predictions('yf')[..., ind_scene]
        w_ = np.exp(_normalize(self.lw, normalize=True))
        frac_thawed = np.zeros((self.ygrid.shape[0],) + w_.shape[:-1])
        # cannot vectorize because of memory issues
        for jy in range(len(self.ygrid)):
            valid = (self.ygrid[jy] < yf)[(np.newaxis,) * len(w_[:-1].shape) + (Ellipsis,)]
            frac_thawed[jy, ...] = np.sum(w_ * valid, axis=-1)
        return frac_thawed

    def expectation(
            self, param='e', etype='mean', p=None, normalize=True, **kwargs):
        if etype == 'mean':
            return self.moment(param=param, p=p, normalize=normalize)
        elif etype in ('var', 'variance'):
            return self.variance(param=param, p=p, normalize=normalize)
        elif etype == 'quantile':
            return self.quantile(kwargs['quantiles'], param=param)
        elif param in ('frac_thawed'):
            return self.frac_thawed(ind_scene=kwargs['ind_scene'])
        else:
            raise NotImplementedError(f"Expectation type {etype} not recognized.")

    def export_expectation(
            self, pathout, param='e', etype='mean', p=None, normalize=True, fn=None, 
            **kwargs):
        res = self.expectation(param=param, etype=etype, p=p, normalize=normalize, **kwargs)
        if fn is None: fn = f'{param}_{etype}.npy'
        fnout = os.path.join(pathout, fn)
        np.save(fnout, res)
        
    def save(self, fnout):
        from analysis import save_object
        dictout = {'geospatial': self.geospatial, 'lw': self.lw, 'predens': self.predens}
        save_object(dictout, fnout)
        
    @classmethod
    def from_file(cls, fn):
        from analysis import load_object
        dictin = load_object(fn)
        ir = InversionResults(**dictin)
        return ir