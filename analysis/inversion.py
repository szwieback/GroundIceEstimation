'''
Created on Sep 14, 2022

@author: simon
'''

from analysis import enforce_directory, MulticlassPredictionEnsemble, ioput

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
    fname = 'fname'  # standard filename

    def __init__(self, predens=None, geospatial=None, batch_size=1024, **kwargs):
        self.predens = predens
        self.geospatial = geospatial
        self.batch_size = batch_size

    def _simulated_observations_single(self, ind_scenes, C_obs=None, ec=None):
        s_pred = self.predens.extract_predictions(
            ind_scenes, C_obs=C_obs, ec=ec, rng=None)  # hardcoded seed for now
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
        if pathout is not None:
            for f in glob.glob(str(self._filename(pathout, self.fname, '*'))):
                try:
                    Path(f).unlink()
                except:
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
    fname = 'lw'
    def _logweights_single(self, ind_scenes, _s_obs, _C_obs, _ec=None, normalize=False):
        from inference import lw_mvnormal, psislw, _normalize
        try:
            if np.count_nonzero(np.isnan(_s_obs)) > 0: raise ValueError('Cannot handle NaN')
            s_pred = self._simulated_observations_single(ind_scenes, ec=_ec)
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
        _fn = self._filename(pathout, self.fname, nbatch)
        if self._overwrite(_fn, overwrite=overwrite):
            n0 = nbatch * self.batch_size
            n1 = min(((nbatch + 1) * self.batch_size, s_obs_flat.shape[-1]))
            _s_obs_batch = s_obs_flat[..., n0:n1].copy()
            _C_obs_batch = C_obs_flat[..., n0:n1].copy()
            lw = []
            for ndiff in range(0, n1 - n0):
                _s, _C = _s_obs_batch[..., ndiff], _C_obs_batch[..., ndiff]
                _ec = ec_flat[n0 + ndiff] if ec_flat is not None else None
                # clean up _logweigths_single [provide predictions; not ind_scenes]
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
            _lw = np.load(self._filename(pathout, self.fname, 0))
            shape = (np.sum(np.array(lw).flatten()), _lw.shape[1])
            fp = np.memmap(fnmmap, dtype=_lw.dtype, mode='w+', shape=shape)
            rm, nrm = 0, 0
            for nbatch in range(Nbatch):
                _lw = np.load(self._filename(pathout, self.fname, nbatch))
                nrm = rm + _lw.shape[0]
                fp[rm:nrm,:] = _lw[:,:]
                rm = nrm
            fp.flush()
            outp = Mmap(fnmmap, _lw.dtype, shape)
        return outp

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
        if 'normalize' not in kwargs: kwargs['normalize'] = False
        _lw = self.inference(
            ind_scenes, s_obs, C_obs, ec=ec, n_jobs=n_jobs, pathout=pathout,
            memory=memory, overwrite=overwrite, **kwargs)
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

class InversionProcessorGM(InversionProcessor):
    fname = 'gmp'
    def __init__(self, predens=None, geospatial=None, batch_size=1024, **kwargs):
        super().__init__(predens=predens, geospatial=geospatial, batch_size=batch_size, **kwargs)
        self.K = kwargs['K'] if 'K' in kwargs else 3
        self.method_condition = kwargs['method_condition'] if 'method_condition' in kwargs else 'unobserved'
        self.variables = kwargs['variables'] if 'variables' in kwargs else ()
        if len(self.variables) == 0:
            import warnings
            warnings.warn("No variables defined in InversionProcessorGM")

    def predicted_variables(self, ec=None):
        unobserved_list = [self._predicted_variable(*v, ec=ec) for v in self.variables]
        unobserved = np.concatenate(unobserved_list, axis=-1)
        return unobserved

    def _predicted_variable(self, param, param_dict={}, ec=None):
        # reverse engineering for gaussian mixture workflow
        if ec is None:
            predens = self.predens
        else:
            predens = self.predens[ec]
        if 'indranges' in param_dict:
            assert 'ind_scene' not in param_dict
            if f'{param}_mean_period' not in predens.results:
                predens.predict_mean_period(param_dict['indranges'], param=param)
            p = predens.results[f'{param}_mean_period']
        elif 'depthranges' in param_dict:
            assert 'ind_scene' not in param_dict
            if f'{param}_mean_depth' not in predens.results:
                predens.predict_mean_depth(param_dict['depthranges'], param=param)
            p = predens.results[f'{param}_mean_depth']            
        else:
            p = predens.results[param]
            if 'ind_scene' in param_dict:
                p = p[:, param_dict['ind_scene']]
        return p

    @property
    def classnames(self):
        try:
            ecs = self.predens.classnames
        except:
            ecs = None
        return ecs

    def marginal(self, ind_scenes):
        classnames = self.classnames
        if classnames is None:
            return self._marginal(ind_scenes, ec=None)
        else:
            return {ec: self._marginal(ind_scenes, ec=ec) for ec in classnames}

    def _marginal(self, ind_scenes, ec=None):
        from inference import GaussianMixtureDistribution
        s_pred = self._simulated_observations_single(ind_scenes, ec=ec)
        unobserved = self.predicted_variables(ec=ec)
        samples = np.concatenate((unobserved, s_pred), axis=-1)
        gmm = GaussianMixtureDistribution.from_samples(samples, K=self.K)
        return gmm

    def posterior(
            self, ind_scenes, s_obs, C_obs, ec=None, n_jobs=8, pathout=None, memory=True, overwrite=False):
        from joblib import Parallel, delayed
        s_obs_flat = np.reshape(s_obs, (s_obs.shape[0], -1))
        C_obs_flat = np.reshape(C_obs, (C_obs.shape[0], C_obs.shape[1], -1))
        ec_flat = ec.flatten() if ec is not None else None
        N = s_obs_flat.shape[-1]
        assert C_obs_flat.shape[-1] == N
        Nbatch = np.int64(np.ceil(N / self.batch_size))
        gmm = self.marginal(ind_scenes)
        def _res(nbatch):
            return self._posterior_batch(
                nbatch, gmm, s_obs_flat, C_obs_flat, ec_flat=ec_flat, memory=memory, pathout=pathout,
                overwrite=overwrite)
        gmp = Parallel(n_jobs=n_jobs)(delayed(_res)(nbatch) for nbatch in range(Nbatch))
        if memory:
            outp = np.concatenate(gmp, axis=0)
        else:
            fnmmap = self._filename(pathout, 'gmpmmap')
            _gmp = np.load(self._filename(pathout, self.fname, 0))
            shape = s_obs.shape[1:] + _gmp.shape[-2:]
            fp = np.memmap(fnmmap, dtype=_gmp.dtype, mode='w+', shape=shape)
            rm, nrm = 0, 0
            for nbatch in range(Nbatch):
                _gmp = np.load(self._filename(pathout, self.fname, nbatch))
                nrm = rm + _gmp.shape[0]
                # reshaping is very awkward
                fp.reshape((-1,) + _gmp.shape[1:])[rm:nrm,:,:] = _gmp[...]  # should be a view
                # check whether it worked
                assert np.sum(np.abs(fp.reshape((-1,) + _gmp.shape[1:])[rm, ...] - _gmp[0, ...])) < 1e-14
                rm = nrm
            fp.flush()
            outp = Mmap(fnmmap, _gmp.dtype, shape)
        return outp

    def _posterior_single(self, gmm, s_obs, C_obs):
        N_nan = np.count_nonzero(np.isnan(s_obs))
        if N_nan > 0:
            ind_invalid = np.any(np.isnan(s_obs), axis=1)
            s_obs[ind_invalid, :] = 0.0
        gmp = gmm.conditional(s_obs, C_obs=C_obs, method_condition=self.method_condition)
        gmp_array = gmp.to_array()
        if N_nan > 0:
            gmp_array[:, ind_invalid, :] = np.nan
        return np.moveaxis(gmp_array, 0, -2)  # so k axis is at -2

    def _posterior_batch(
            self, nbatch, gmm, s_obs_flat, C_obs_flat, ec_flat=None, pathout=None, memory=True,
            overwrite=False):
        _fn = self._filename(pathout, self.fname, nbatch)
        if self._overwrite(_fn, overwrite=overwrite):
            n0 = nbatch * self.batch_size
            n1 = min(((nbatch + 1) * self.batch_size, s_obs_flat.shape[-1]))
            _s_obs_batch = np.moveaxis(s_obs_flat[..., n0:n1], -1, 0).copy()
            _C_obs_batch = np.moveaxis(C_obs_flat[..., n0:n1], -1, 0).copy()
            _ec_batch = ec_flat[n0:n1].copy() if ec_flat is not None else None
            if _ec_batch is not None:
                gmp_arr = None
                for cn in self.classnames:
                    ind = np.nonzero(_ec_batch == cn)[0]
                    if len(ind) > 0:
                        _s_obs_batch_cn = _s_obs_batch[ind, ...]
                        _C_obs_batch_cn = _C_obs_batch[ind, ...]
                        gmp_arr_cn = self._posterior_single(gmm[cn], _s_obs_batch_cn, _C_obs_batch_cn)
                    if gmp_arr is None:
                        gmp_arr = np.zeros((n1-n0,) + gmp_arr_cn.shape[1:], dtype=gmp_arr_cn.dtype)
                    gmp_arr[ind, ...] = gmp_arr_cn
            else:
                gmp_arr = self._posterior_single(gmm, _s_obs_batch, _C_obs_batch)  # array
            if _fn is not None:
                enforce_directory(_fn)
                np.save(_fn, gmp_arr)
        else:
            gmp_arr = np.load(_fn)
        outp = gmp_arr if memory else gmp_arr.shape[1]
        return outp

    def inference(
            self, ind_scenes, s_obs, C_obs, ec=None, n_jobs=8, pathout=None, memory=True, overwrite=False,
            **kwargs):
        return self.posterior(
            ind_scenes, s_obs, C_obs, ec=ec, n_jobs=n_jobs, pathout=pathout, memory=memory,
            overwrite=overwrite)

    def results(
            self, ind_scenes, s_obs, C_obs, ec=None, n_jobs=8, pathout=None, memory=True,
            overwrite=False, **kwargs):
        _gmp = self.inference(
            ind_scenes, s_obs, C_obs, ec=ec, n_jobs=n_jobs, pathout=pathout, memory=memory,
            overwrite=overwrite)
        shape = s_obs.shape[1:] + (_gmp.shape[-2],) + (_gmp.shape[-1],)
        if memory:
            if ec is None:
                return InversionResultsGM(
                    self.predens, np.reshape(_gmp, shape), geospatial=self.geospatial,
                    variables=self.variables)
            else:
                return MulticlassInversionResultsGM(
                    self.predens, np.reshape(_gmp, shape), ec, geospatial=self.geospatial, 
                    variables=self.variables)
        else:
            mmap = Mmap(_gmp.filename, _gmp.dtype, shape)
            if ec is None:
                return InversionResultsGMMmap(
                    self.predens, mmap, geospatial=self.geospatial, variables=self.variables)
            else:
                return MulticlassInversionResultsGMMmap(
                    self.predens, mmap, ec, geospatial=self.geospatial, variables=self.variables)

class InversionResults():
    # abstract class
    blocksize_default = 1024
    intdims = 1  # internal dimensions
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
            self, pathout, param='e', etype='mean', p=None, fn=None, hdf5=False, overwrite=True, **kwargs):
        if fn is None: fn = f'{param}_{etype}.npy'
        fnout = pathout / fn
        if not fnout.exists() or overwrite:
            res = self.expectation(param=param, etype=etype, p=p, **kwargs)
            np.save(fnout, res)
        else:
            res = np.load(fnout)
        if hdf5:
            fnhdf5 = fnout.with_suffix('.h5')
            self._export_hdf5(res, fnhdf5, param=param, etype=etype)

    def _export_hdf5(
            self, res, fnh5, param='e', etype='mean'):
        
        dataset_name = f'{param}_{etype}'
        attrs = self.geospatial._hdf5_attributes()
        if dataset_name in (
            'e_mean_period_var', 'e_mean_period_mean', 'e_mean_depth_mean', 'e_mean_depth_var'):
            if res.ndim == 3 and res.shape[2] == 1:
                res = res[:, :, 0]
            ioput.save_hdf5(res, attrs, dataset_name, fnh5)
        # elif dataset_name in ('e_mean_period_quantile', 'e_mean_depth_quantile'):
        #     if res.ndim == 4 and res.shape[2] == 1:
        #         res = res[:, :, 0, :]
        #     ioput.save_hdf5(res, attrs, dataset_name, fnh5)
        elif dataset_name in ('e_mean', 'e_var', 'frac_thawed_None'):
            layer_name = 'depth_mm'
            dlist = self._depth_mm_list
            ioput.save_hdf5(res, attrs, dataset_name, fnh5, layer_name=layer_name, layer_info=dlist)
        elif dataset_name in ('yf_mean'):
            layer_name = 'dates'
            dtlist = self._dt_strlist
            ioput.save_hdf5(res, attrs, dataset_name, fnh5, layer_name=layer_name, layer_info=dtlist)
        else:
            import warnings
            warnings.warn(f"Data type {dataset_name} H5 output not implemented")

    def _expectation(self, param='e', etype='mean', p=None, **kwargs):
        if etype == 'mean':
            return self._mean(param=param, p=p, **kwargs)
        elif etype in ('var', 'variance'):
            return self._variance(param=param, p=p, **kwargs)
        elif etype == 'quantile':
            q = kwargs.pop('quantiles')
            return self._quantile(q, param=param, **kwargs)
        elif param in ('frac_thawed'):
            ind_scene = kwargs.pop('ind_scene')
            return self._frac_thawed(ind_scene=ind_scene, **kwargs)
        else:
            raise NotImplementedError(f"Expectation type {etype} not recognized.")

    @abstractmethod
    def _moment(self, param='e', power=1, p=None, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _variance(self, param='e', p=None, **kwargs):
        raise NotImplementedError()

    def _mean(self, param='e', p=None, **kwargs):
        return self._moment(param=param, power=1, p=p, **kwargs)

    def _invres_generator(self, block_size=None):
        if block_size is None:
            block_size = self.blocksize
        iss = self.invres.shape
        # maybe reshape to make this more general
        step = block_size if len(iss) == 2 else np.prod(iss[1:-self.intdims]) // block_size
        ind = np.arange(self.invres.shape[0], step=max((1, step)))[1:]
        for _invres in np.array_split(self.invres, ind, axis=0):
            yield _invres  # view to avoid memory issues

    def register_dates(self, datelist):
        self.dates = datelist
        
    @property
    def _dt_strlist(self):
        return self.dates.strftime('%Y-%m-%d').tolist()

    @property
    def _depth_mm_list(self):
        return [f'{int(1000*y)}-{int(1000*(y + self.dy))}' for y in self.ygrid]

    def _filename(self, path0, ftype, number=None, ext='npy'):
        _fn = ftype if number is None else f'{ftype}_{number}'
        return path0 / f'{_fn}.{ext}'

class InversionResultsIS(InversionResults):

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
                Parallel(n_jobs=n_jobs)(delayed(fun)(_lw) for _lw in self._invres_generator(block_size)),
                axis=0)
            return res

    def __frac_thawed(self, ind_scene, _lw, normalize=True):
        from inference import _normalize
        yf = self.predictions('yf')[..., ind_scene]
        w_ = np.exp(_normalize(_lw, normalize=normalize))
        frac_thawed = np.zeros((self.ygrid.shape[0],) + w_.shape[:-1])
        # cannot vectorize because of memory issues
        for jy in range(len(self.ygrid)):
            valid = (self.ygrid[jy] < yf)[(np.newaxis,) * len(w_[:-1].shape) + (Ellipsis,)]
            frac_thawed[jy, ...] = np.sum(w_ * valid, axis=-1)
        return np.moveaxis(frac_thawed, 0, -1)

    def _frac_thawed(self, ind_scene, normalize=True, n_jobs=-1):
        def _ft(_lw):
            return self.__frac_thawed(ind_scene, _lw, normalize=normalize)
        return self._parallel(_ft, n_jobs=n_jobs)

    def _moment(self, param='e', power=1, p=None, normalize=True, n_jobs=-1):
        from inference import expectation
        p = self.predictions(param=param, p=p)
        def __moment(_lw):
            _m = expectation(np.power(p, power), _lw, normalize=normalize)
            return _m
        mom = self._parallel(__moment, n_jobs=n_jobs)
        return mom

    def _variance(self, param='e', p=None, normalize=True, **kwargs):
        # improvement needed to deal with numerical issues
        p = self.predictions(param=param, p=p)
        var = (self._moment(param=None, p=p, power=2, normalize=normalize, **kwargs)
               -self._moment(param=None, p=p, power=1, normalize=normalize, **kwargs) ** 2)
        return var

    def _quantile(
            self, quantiles, param='e', smooth=None, steps=8, p=None, method='bisection', n_jobs=-1, 
            **kwargs):
        from inference import quantile as quant
        p = self.predictions(param=param, p=p)
        def __quantile(_lw):
            _pq = quant(
                p, _lw, quantiles, method=method, steps=steps, normalize=True,
                smooth=smooth)
            return _pq
        postquant = self._parallel(__quantile, n_jobs=n_jobs)
        return np.reshape(postquant, self.lw.shape[0:-1] + postquant.shape[1:])

class InversionResultsGM(InversionResults):
    intdims = 2  # internal dimensions

    def __init__(self, predens, invres, geospatial=None, blocksize=None, variables=None):
        super().__init__(predens, invres, geospatial=geospatial, blocksize=blocksize)
        self.variables = variables

    def _indices_variables(self, param='e'):
        def l(v):
            _l = 1
            if 'indranges' in v[1]: _l = len(v[1]['indranges'])
            if 'depthranges' in v[1]: _l = len(v[1]['depthranges'])
            if 'ind_scene' in v[1]: _l = len(v[1]['ind_scene'])
            return _l
        n_variables = np.array([l(v) for v in self.variables])
        def _name_post(v):
            if 'indranges' in v[1]: 
                np = v[0] + '_mean_period'
            elif 'depthranges' in v[1]:
                np = v[0] + '_mean_depth'
            else:
                np = v[0]
            return np
        variables_post = [_name_post(v) for v in self.variables]
        ind_v = variables_post.index(param)
        st = np.sum(n_variables[:ind_v])
        return np.arange(st, st + n_variables[ind_v], dtype=np.int64)

    @property
    def gmp(self):
        from inference import GaussianMixtureDistribution
        return GaussianMixtureDistribution.from_array(np.moveaxis(self.invres, -2, 0))

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'invres': self.invres, 'predens': self.predens,
            'blocksize': self.blocksize, 'variables': self.variables}
        return dictout

    def expectation(self, param='e', etype='mean', p=None, **kwargs):
        return self._expectation(param=param, etype=etype, p=p, **kwargs)

    def _parallel(self, fun, n_jobs=-1, block_size=None):
        if n_jobs in (0, 1, None):
            return fun(self.lw)
        else:
            from joblib import Parallel, delayed
            res = np.concatenate(
                Parallel(n_jobs=n_jobs)(delayed(fun)(_lw) for _lw in self._invres_generator(block_size)),
                axis=0)
            return res

    def _mean(self, param='e', p=None, **kwargs):
        if p is not None: raise NotImplementedError()
        indices = self._indices_variables(param=param)
        return self.gmp.mean(indices)

    def _variance(self, param='e', p=None, **kwargs):
        if p is not None: raise NotImplementedError()
        indices = self._indices_variables(param=param)
        return self.gmp.variance(indices)
    
    def _quantile(self, quantiles, param='e', p=None, **kwargs):
        if p is not None: raise NotImplementedError()
        indices = self._indices_variables(param=param)
        return self.gmp.quantile(quantiles, indices, **kwargs)

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
        _invres = self.invres[ind, ...]
        return InversionResultsIS(self.predens[cn], _invres, blocksize=self.blocksize)

    def expectation(self, param='e', etype='mean', p=None, **kwargs):
        res = None
        if p is not None: raise ValueError('p input not supported')
        for cn in self.predens.classnames:
            ir, ind = self[cn], (self.ec.flatten() == cn)
            res_cn = ir._expectation(param=param, etype=etype, p=None, **kwargs)
            if res is None:
                res = np.full(
                    (np.prod(self.invres.shape[:-1]),) + res_cn.shape[1:], np.nan, dtype=res_cn.dtype)
            res[ind, ...] = res_cn
        res = np.reshape(res, self.invres.shape[:-1] + res.shape[1:])
        return res

class MulticlassInversionResultsGM(InversionResultsGM):

    def __init__(self, predens, invres, ec, geospatial=None, blocksize=None, variables=None):
        super().__init__(predens, invres, geospatial=geospatial, blocksize=blocksize, variables=variables)
        if not issubclass(type(predens), MulticlassPredictionEnsemble):
            raise ValueError("Prediction ensemble incompatible with MuticlassInversionResults")
        self.ec = ec

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'invres': self.invres, 'predens': self.predens,
            'blocksize': self.blocksize, 'ec': self.ec, 'variables': self.variables}
        return dictout

    def __getitem__(self, cn):
        ind = (self.ec == cn)
        _invres = self.invres[ind, ...]
        return InversionResultsGM(
            self.predens[cn], _invres, blocksize=self.blocksize, variables=self.variables)


class InversionResultsGMMmap(InversionResultsGM):

    def __init__(self, predens, gmpmmap, geospatial=None, blocksize=None, variables=None, temporary=False):
        InversionResultsGM.__init__(
            self, predens, None, geospatial=geospatial, blocksize=blocksize, variables=variables)
        if gmpmmap is not None:
            self.gmpmmap = gmpmmap
            self.invres = np.memmap(gmpmmap.filename, dtype=gmpmmap.dtype, mode='r', shape=gmpmmap.shape)
        self.temporary = temporary

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'gmpmmap': self.gmpmmap, 'predens': self.predens,
            'blocksize': self.blocksize, 'variables': self.variables}
        return dictout

    def __del__(self):
        if self.temporary:
            try:
                Path(self.gmpmmap.filename).unlink()
            except:
                pass

    @staticmethod
    def _dict_from_file(fn):
        from analysis import load_object
        dictin = load_object(fn)
        if not Path(dictin['gmpmmap'].filename).exists:
            dictin['gmpmmap'] = None
        return dictin

    @classmethod
    def from_file(cls, fn):
        return cls(**InversionResultsGMMmap._dict_from_file(fn))

class InversionResultsISMmap(InversionResultsIS):
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
        try:
            if self.temporary:
                Path(self.lwmmap.filename).unlink()
        except:
            pass

    @staticmethod
    def _dict_from_file(fn):
        from analysis import load_object
        dictin = load_object(fn)
        if not Path(dictin['lwmmap'].filename).exists():
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
            self.lwmmap = lwmmap
            if Path(lwmmap.filename).exists():
                self.invres = np.memmap(lwmmap.filename, dtype=lwmmap.dtype, mode='r', shape=lwmmap.shape)
        self.temporary = temporary

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'lwmmap': self.lwmmap, 'predens': self.predens,
            'blocksize': self.blocksize, 'ec': self.ec}
        return dictout


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


class MulticlassInversionResultsGMMmap(MulticlassInversionResultsGM):

    def __init__(
            self, predens, gmpmmap, ec, geospatial=None, blocksize=None, temporary=False, variables=None):
        MulticlassInversionResultsGM.__init__(
            self, predens, None, ec, geospatial=geospatial, blocksize=blocksize, variables=variables)
        if gmpmmap is not None:
            self.gmpmmap = gmpmmap
            if Path(gmpmmap.filename).exists():
                self.invres = np.memmap(
                    gmpmmap.filename, dtype=gmpmmap.dtype, mode='r', shape=gmpmmap.shape)
        self.temporary = temporary

    @property
    def _dict(self):
        dictout = {
            'geospatial': self.geospatial, 'gmpmmap': self.gmpmmap, 'predens': self.predens,
            'blocksize': self.blocksize, 'ec': self.ec, 'variables': self.variables}
        return dictout

    def __getitem__(self, cn):
        raise NotImplementedError() # not needed
