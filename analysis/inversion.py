'''
Created on Sep 14, 2022

@author: simon
'''

from analysis import enforce_directory, MulticlassPredictionEnsemble

import numpy as np
from pathlib import Path
from collections import namedtuple
from abc import abstractmethod
import glob

Mmap = namedtuple('Mmap', ('filename', 'dtype', 'shape'))

def thaw_depth(frac_thawed, ygrid, frac=0.5, return_indices=False):
    if len(frac_thawed.shape) > 2:
        ft = np.reshape(frac_thawed, (-1, frac_thawed.shape[-1]))
        td = thaw_depth(ft, ygrid, frac=frac)
        td = np.reshape(td, frac_thawed.shape[:-1])
    elif len(frac_thawed.shape) == 2:
        ind = np.argmax((frac_thawed < frac), axis=1)
        td = np.take_along_axis(ygrid, ind, axis=0)
    else:
        ind = np.nonzero(frac_thawed < frac)[0][0]
        td = ygrid[ind]
    if return_indices:
        return ind
    else:
        return td

class InversionProcessor():
    # abstract class
    # uses the same ensemble but different C_obs
    def __init__(self, predens=None, geospatial=None, batch_size=1024):
        self.predens = predens
        self.geospatial = geospatial
        self.batch_size = batch_size

    def _simulated_observations_single(self, ind_scenes, _C_obs, ec=None):
        s_pred = self.predens.extract_predictions(
            ind_scenes, C_obs=_C_obs, ec=ec, rng=None)  # hardcoded seed for now
        return s_pred

    def _filename(self, path0, ftype, number=None, ext='npy'):
        if path0 is None:
            return None
        else:
            _fn = ftype if number is None else f'{ftype}_{number}'
            return path0 / f'{_fn}.{ext}'

    def _overwrite(self, fn, overwrite=False):
        try:
            ow = not fn.exists() or overwrite
        except:
            ow = True
        return ow

    @abstractmethod
    def inference(self, ind_scenes, s_obs, C_obs, ec=None, n_jobs=8, pathout=None,
                  memory=True, overwrite=False, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def results(
            self, ind_scenes, s_obs, C_obs, ec=None, n_jobs=8, pathout=None, memory=True,
            overwrite=False, **kwargs):
        raise NotImplementedError()

    def delete_temporary(self, pathout):
        pass

    @property
    def depth(self):
        return self.predens.depth

    @property
    def dy(self):
        return self.predens.dy

    @property
    def ygrid(self):
        return self.predens.ygrid

class InversionProcessorIS(InversionProcessor):
    def _logweights_single(self, ind_scenes, _s_obs, _C_obs, _ec=None, normalize=False):
        from inference import lw_mvnormal, psislw, _normalize
        try:
            s_pred = self._simulated_observations_single(ind_scenes, _C_obs, ec=_ec)
            if np.count_nonzero(np.isnan(_s_obs)) > 0: raise ValueError('Cannot handle NaN')
            lw = lw_mvnormal(
                _s_obs[np.newaxis,:], _C_obs[np.newaxis, ...], s_pred)
            lw_ps, _ = psislw(lw)
            lw_ps = _normalize(lw_ps, normalize=normalize)
        except:
            lw_ps = np.full((1, self.predens.N), np.nan)
        return lw_ps

    def _logweights_batch(
            self, nbatch, ind_scenes, s_obs_flat, C_obs_flat, ec_flat=None, normalize=False, pathout=None,
            memory=True, overwrite=False):
        _fn = self._filename(pathout, 'lw', nbatch)
        if self._overwrite(_fn, overwrite=overwrite):
            n0 = nbatch * self.batch_size
            n1 = min(((nbatch + 1) * self.batch_size, s_obs_flat.shape[-1]))
            _s_obs_batch = s_obs_flat[..., n0:n1].copy()
            _C_obs_batch = C_obs_flat[..., n0:n1].copy()
            lw = []
            for ndiff in range(0, n1 - n0):
                _s, _C = _s_obs_batch[..., ndiff], _C_obs_batch[..., ndiff]
                _ec = ec_flat[n0 + ndiff] if ec_flat is not None else None
                lw.append(self._logweights_single(ind_scenes, _s, _C, _ec=_ec, normalize=normalize))
            lw = np.concatenate(lw, axis=0)
            if _fn is not None:
                enforce_directory(_fn)
                np.save(_fn, lw)
        else:
            lw = np.load(_fn)
        outp = lw if memory else len(lw)
        return outp

    def logweights(
            self, ind_scenes, s_obs, C_obs, ec=None, n_jobs=8, normalize=False, pathout=None, memory=True,
            overwrite=False):
        from joblib import Parallel, delayed
        s_obs_flat = np.reshape(s_obs, (s_obs.shape[0], -1))
        C_obs_flat = np.reshape(C_obs, (C_obs.shape[0], C_obs.shape[1], -1))
        ec_flat = ec.flatten() if ec is not None else None
        N = s_obs_flat.shape[-1]
        assert C_obs_flat.shape[-1] == N
        Nbatch = np.int64(np.ceil(N / self.batch_size))
        def _res(nbatch):
            return self._logweights_batch(
                nbatch, ind_scenes, s_obs_flat, C_obs_flat, ec_flat=ec_flat, normalize=normalize,
                memory=memory, pathout=pathout, overwrite=overwrite,)
        lw = Parallel(n_jobs=n_jobs)(delayed(_res)(nbatch) for nbatch in range(Nbatch))
        if memory:
            outp = np.concatenate(lw, axis=0)
        else:
            fnmmap = self._filename(pathout, 'lwmmap')
            _lw = np.load(self._filename(pathout, 'lw', 0))
            shape = (np.sum(np.array(lw).flatten()), _lw.shape[1])
            fp = np.memmap(fnmmap, dtype=_lw.dtype, mode='w+', shape=shape)
            rm, nrm = 0, 0
            for nbatch in range(Nbatch):
                _lw = np.load(self._filename(pathout, 'lw', nbatch))
                nrm = rm + _lw.shape[0]
                fp[rm:nrm,:] = _lw[:,:]
                rm = nrm
            fp.flush()
            outp = Mmap(fnmmap, _lw.dtype, shape)
        return outp

    def delete_temporary(self, pathout):
        if pathout is not None:
            for f in glob.glob(str(self._filename(pathout, 'lw', '*'))):
                try:
                    Path(f).unlink()
                except:
                    pass

    def inference(
            self, ind_scenes, s_obs, C_obs, ec=None, n_jobs=8, pathout=None, memory=True, overwrite=False,
            **kwargs):
        _n = kwargs['normalize'] if 'normalize' in kwargs else False
        return self.logweights(
            ind_scenes, s_obs, C_obs, ec=ec, n_jobs=n_jobs, normalize=_n, pathout=pathout, memory=memory,
            overwrite=overwrite)

    def results(
            self, ind_scenes, s_obs, C_obs, ec=None, n_jobs=8, pathout=None, memory=True,
            overwrite=False, **kwargs):
        normalize = kwargs['normalize'] if 'normalize' in kwargs else False
        _lw = self.logweights(
            ind_scenes, s_obs, C_obs, ec=ec, n_jobs=n_jobs, normalize=normalize, pathout=pathout,
            memory=memory, overwrite=overwrite)
        shape = s_obs.shape[1:] + (_lw.shape[-1],)
        if memory:
            if ec is None:
                return InversionResultsIS(self.predens, np.reshape(_lw, shape), geospatial=self.geospatial)
            else:
                return MulticlassInversionResultsIS(
                    self.predens, np.reshape(_lw, shape), ec, geospatial=self.geospatial)
        else:
            # these two should be equivalent, hence simply overwrite tuple
            # lw = np.memmap(_lw.filename, dtype=_lw.dtype, mode='r', shape=_lw.shape).reshape(shape)
            # _lw = np.memmap(_lw.filename, dtype=_lw.dtype, mode='r', shape=shape)
            mmap = Mmap(_lw.filename, _lw.dtype, shape)
            if ec is None:
                return InversionResultsISMmap(self.predens, mmap, geospatial=self.geospatial)
            else:
                return MulticlassInversionResultsISMmap(self.predens, mmap, ec, geospatial=self.geospatial)

class InversionResults():
    # abstract class
    blocksize_default = 1024
    def __init__(self, predens, invres, geospatial=None, blocksize=None):
        self.predens = predens
        self.invres = invres
        self.geospatial = geospatial
        self.blocksize = blocksize if blocksize is not None else InversionResults.blocksize_default

    @property
    def depth(self):
        return self.predens.depth

    @property
    def dy(self):
        return self.predens.dy

    @property
    def ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

    def predictions(self, param='e', p=None):
        if p is None:
            p = self.predens.results[param]
        return p

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'invres': self.invres, 'predens': self.predens,
            'blocksize': self.blocksize}
        return dictout

    def save(self, fnout):
        from analysis import save_object
        save_object(self._dict, fnout)

    @classmethod
    def from_file(cls, fn):
        from analysis import load_object
        dictin = load_object(fn)
        ir = cls(**dictin)
        return ir

    @abstractmethod
    def expectation(self, param='e', etype='mean', p=None, **kwargs):
        raise NotImplementedError()

    def export_expectation(
            self, pathout, param='e', etype='mean', p=None, fn=None, **kwargs):
        res = self.expectation(param=param, etype=etype, p=p, **kwargs)
        if fn is None: fn = f'{param}_{etype}.npy'
        fnout = pathout / fn
        np.save(fnout, res)

    def _expectation(self, param='e', etype='mean', p=None, normalize=True, **kwargs):
        if etype == 'mean':
            return self._moment(param=param, p=p, normalize=normalize)
        elif etype in ('var', 'variance'):
            return self._variance(param=param, p=p, normalize=normalize)
        elif etype == 'quantile':
            return self._quantile(kwargs['quantiles'], param=param)
        elif param in ('frac_thawed'):
            return self._frac_thawed(ind_scene=kwargs['ind_scene'])
        else:
            raise NotImplementedError(f"Expectation type {etype} not recognized.")

    @abstractmethod
    def _moment(self, param='e', power=1, p=None, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def _variance(self, param='e', p=None, **kwargs):
        raise NotImplementedError()

class InversionResultsIS(InversionResults):
    def __init__(self, predens, invres, geospatial=None, blocksize=None):
        super().__init__(predens, invres, geospatial=geospatial, blocksize=blocksize)

    @property
    def lw(self):
        return self.invres

    def expectation(self, param='e', etype='mean', p=None, **kwargs):
        normalize = kwargs['normalize'] if 'normalize' in kwargs else True
        return self._expectation(param=param, etype=etype, p=p, normalize=normalize, **kwargs)

    def _parallel(self, fun, n_jobs=-1, block_size=None):
        if n_jobs in (0, 1, None):
            return fun(self.lw)
        else:
            from joblib import Parallel, delayed
            res = np.concatenate(
                Parallel(n_jobs=n_jobs)(delayed(fun)(_lw) for _lw in self._lw_generator(block_size)),
                axis=0)
            return res

    def __frac_thawed(self, ind_scene, _lw):
        from inference import _normalize
        yf = self.predictions('yf')[..., ind_scene]
        w_ = np.exp(_normalize(_lw, normalize=True))
        frac_thawed = np.zeros((self.ygrid.shape[0],) + w_.shape[:-1])
        # cannot vectorize because of memory issues
        for jy in range(len(self.ygrid)):
            valid = (self.ygrid[jy] < yf)[(np.newaxis,) * len(w_[:-1].shape) + (Ellipsis,)]
            frac_thawed[jy, ...] = np.sum(w_ * valid, axis=-1)
        return np.moveaxis(frac_thawed, 0, -1)

    def _frac_thawed(self, ind_scene, n_jobs=-1):
        def _ft(_lw):
            return self.__frac_thawed(ind_scene, _lw)
        return self._parallel(_ft, n_jobs=n_jobs)

    def _moment(self, param='e', power=1, p=None, normalize=True, n_jobs=-1):
        from inference import expectation
        p = self.predictions(param=param, p=p)
        def __moment(_lw):
            _m = expectation(np.power(p, power), _lw, normalize=normalize)
            return _m
        mom = self._parallel(__moment, n_jobs=n_jobs)
        return mom

    def _variance(self, param='e', p=None, normalize=True):
        # improvement needed to deal with numerical issues
        p = self.predictions(param=param, p=p)
        var = (self._moment(param=None, p=p, power=2, normalize=normalize)
               -self._moment(param=None, p=p, power=1, normalize=normalize) ** 2)
        return var

    def _quantile(
            self, quantiles, param='e', smooth=None, steps=8, p=None, method='bisection',
            n_jobs=-1):
        from inference import quantile as quant
        p = self.predictions(param=param, p=p)
        def __quantile(_lw):
            _pq = quant(
                p, _lw, quantiles, method=method, steps=steps, normalize=True,
                smooth=smooth)
            return _pq
        postquant = self._parallel(__quantile, n_jobs=n_jobs)
        return np.reshape(postquant, self.lw.shape[0:-1] + postquant.shape[1:])

    def _lw_generator(self, block_size=None):
        if block_size is None:
            block_size = self.blocksize
        step = block_size if len(self.lw.shape) == 2 else np.product(self.lw.shape[1:-1]) // block_size
        ind = np.arange(self.lw.shape[0], step=max((1, step)))[1:]
        for _lw in np.array_split(self.lw, ind, axis=0):
            yield _lw  # view to avoid memory issues

class MulticlassInversionResultsIS(InversionResultsIS):

    def __init__(self, predens, invres, ec, geospatial=None, blocksize=None):
        super().__init__(predens, invres, geospatial=geospatial, blocksize=blocksize)
        if not issubclass(type(predens), MulticlassPredictionEnsemble):
            raise ValueError("Prediction ensemble incompatible with MuticlassInversionResults")
        self.ec = ec

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'invres': self.invres, 'predens': self.predens,
            'blocksize': self.blocksize, 'ec': self.ec}
        return dictout

    def __getitem__(self, cn):
        ind = (self.ec == cn)
        _lw = self.lw[ind, ...]
        return InversionResultsIS(self.predens[cn], _lw, blocksize=self.blocksize)

    def expectation(self, param='e', etype='mean', p=None, **kwargs):
        res = None
        if p is not None: raise ValueError('p input not supported')
        for cn in self.predens.classnames:
            ir, ind = self[cn], (self.ec.flatten() == cn)
            res_cn = ir._expectation(param=param, etype=etype, p=None, **kwargs)
            if res is None:
                res = np.empty((np.product(self.lw.shape[:-1]),) + res_cn.shape[1:], dtype=res_cn.dtype)
            res[ind, ...] = res_cn
        res = np.reshape(res, self.lw.shape[:-1] + res.shape[1:])
        return res

class InversionResultsISMmap(InversionResults):

    def __init__(self, predens, lwmmap, geospatial=None, blocksize=None, temporary=False):
        InversionResultsIS.__init__(self, predens, None, geospatial=geospatial, blocksize=blocksize)
        if lwmmap is not None:
            self.lwmmap = lwmmap
            self.invres = np.memmap(lwmmap.filename, dtype=lwmmap.dtype, mode='r', shape=lwmmap.shape)
        self.temporary = temporary

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'lwmmap': self.lwmmap, 'predens': self.predens,
            'blocksize': self.blocksize}
        return dictout

    def __del__(self):
        if self.temporary:
            try:
                Path(self.lwmmap.filename).unlink()
            except:
                pass

    @staticmethod
    def _dict_from_file(fn):
        from analysis import load_object
        dictin = load_object(fn)
        if not Path(dictin['lwmmap'].filename).exists:
            dictin['lwmmap'] = None
        return dictin

    @classmethod
    def from_file(cls, fn):
        return cls(**InversionResultsISMmap._dict_from_file(fn))

class MulticlassInversionResultsISMmap(MulticlassInversionResultsIS):

    def __init__(self, predens, lwmmap, ec, geospatial=None, blocksize=None, temporary=False):
        MulticlassInversionResultsIS.__init__(
            self, predens, None, ec, geospatial=geospatial, blocksize=blocksize)
        if lwmmap is not None:
            lwmmap = lwmmap
            self.lwmmap = lwmmap
            if Path(lwmmap.filename).exists():
                self.invres = np.memmap(lwmmap.filename, dtype=lwmmap.dtype, mode='r', shape=lwmmap.shape)
        self.temporary = temporary

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'invres': self.lwmmap, 'predens': self.predens,
            'blocksize': self.blocksize, 'ec': self.ec}
        return dictout

    def _filename(self, path0, ftype, number=None, ext='npy'):
        _fn = ftype if number is None else f'{ftype}_{number}'
        return path0 / f'{_fn}.{ext}'

    def __getitem__(self, cn):
        ind = (self.ec == cn)
        shape = (np.count_nonzero(ind), self.lw.shape[-1])
        fnmmap = self._filename(self.lwmmap.filename.parent, 'lwmmap', cn)
        mmap = Mmap(fnmmap, self.lwmmap.dtype, shape)
        fp = np.memmap(mmap.filename, dtype=mmap.dtype, mode='w+', shape=mmap.shape)
        ncum = 0
        for jrow, _ind in enumerate(ind):  # loop to reduce memory footprint
            _n = ncum + np.count_nonzero(_ind)
            fp[ncum:_n, ...] = self.lw[jrow, _ind, ...]
            ncum = _n
        fp.flush()
        del fp
        ir = InversionResultsISMmap(
            self.predens[cn], mmap, blocksize=self.blocksize, temporary=True)
        return ir

    @classmethod
    def from_file(cls, fn):
        return cls(**InversionResultsISMmap._dict_from_file(fn))

