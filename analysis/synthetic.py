'''
Created on Aug 11, 2021

@author: simon
'''
import numpy as np
from pathlib import Path
from abc import abstractmethod
from analysis import save_object, load_object

class InversionSimulator():
    def __init__(self, predens=None, predens_sim=None, rng=None):
        self.predens = predens
        self.predens_sim = predens_sim
        self.rng = rng if rng is not None else np.random.default_rng(seed=1)
        self.ind_scenes = []
        self.C_obs = np.ones((1, 1))

    @classmethod
    def from_file(cls, fn):
        isdict = load_object(fn)
        invsim = cls(
            predens=isdict['predens'], predens_sim=isdict['predens_sim'], rng=isdict['rng'])
        if 'ind_scenes' in isdict:
            invsim.register_observations(isdict['ind_scenes'], isdict['C_obs'])
        if 'variables' in isdict:
            invsim.register_variables(isdict['variables'])
        return invsim

    def register_observations(self, ind_scenes, C_obs):
        self.ind_scenes = ind_scenes
        self.C_obs = C_obs
        
    def register_variables(self, variables=None):
        # not needed for IS but needed for GM
        self.variables = variables
        
    @property
    def predictions_scenes(self):
        s_pred = self.predens.extract_predictions(
            self.ind_scenes, C_obs=None)
        return s_pred

    @property
    def referenced_truth(self):
        return self.predens_sim.extract_predictions(
            self.ind_scenes, C_obs=None, reference_only=True)

    @property
    def depth(self):
        return self.predens.depth

    @property
    def dy(self):
        return self.predens.dy

    @property
    def ygrid(self):
        return self.predens.ygrid

    def simulated_observations(self, rng=None):
        rng_ = self.rng if rng is None else np.random.default_rng(rng)
        return self.predens_sim.extract_predictions(
            self.ind_scenes, C_obs=self.C_obs, rng=rng_)

    def export(self, fnout):
        isdict = {
            'predens': self.predens, 'predens_sim': self.predens_sim, 'rng': self.rng}
        try:
            isdict['ind_scenes'] = self.ind_scenes
            isdict['C_obs'] = self.C_obs
        except:
            pass
        try:
            isdict['variables'] = self.variables
        except:
            pass
        save_object(isdict, fnout)

    def _replicate_generator(self, pathout, replicates=None):
        if replicates is not None:
            for r in replicates:
                yield r
        else:
            from itertools import count
            for r in count(0):
                if self.filename_sim(pathout, r).exists():
                    yield r
                else:
                    break

    def prescribed(self, param='e'):
        return self.predens_sim.results[param]

    def prescribed_mean_period(self, indranges, param='e'):
        ref = self.prescribed(param)
        yf = self.prescribed('yf')
        ref_mean = self._mean_period(ref, indranges, yf)
        return ref_mean

    def _mean_period(self, p, indranges, yf):
        # need to re-organize this to avoid redundancy
        ygrid = self.ygrid
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

    def filename_sim(self, pathout, r):
        return pathout / f'sim_{r}.npy'

    def filename_simobs(self, pathout, r):
        return pathout / f'simobs_{r}.npy'

    def filename_metrics(self, pathout, r, suffix=None):
        if suffix is not None:
            suffix_ = '_' + '_'.join(suffix)
        else:
            suffix_ = ''
        rstr = '' if r is None else f'_{r}'
        return pathout / f'metrics{suffix_}{rstr}.p'

    def _suffix(self, param, indranges, prior):
        suffix = (param,) if indranges is None else (param, 'indranges')
        if prior: suffix = suffix + ('prior',)
        return suffix

    @abstractmethod
    def inference(self, replicates=10, pathout=None, n_jobs=-1):
        raise NotImplementedError()

    @abstractmethod
    def results(self, pathout, prior=False, replicates=None):
        raise NotImplementedError()

    def export_metrics(
            self, pathout, param='e', metrics_ind=None, metrics=None, indranges=None,
            prior=False, n_jobs=-1):
        def _export(r):
            sie = self.results(pathout, replicates=(r,), prior=prior)
            suffix = self._suffix(param, indranges, prior)
            fnout = self.filename_metrics(pathout, r, suffix=suffix)
            sie.export_metrics(fnout, param=param, metrics=metrics_ind, indranges=indranges)
        from joblib import Parallel, delayed
        rgen = self._replicate_generator(pathout, replicates=self._replicates(prior))
        Parallel(n_jobs=n_jobs)(delayed(_export)(r) for r in rgen)
        self._assemble_metrics(
            pathout, param='e', metrics=metrics, delete_temp=True, indranges=indranges,
            prior=prior)

    def _validation_metric(self, metrics_dict, metric):
        if metric[0] == 'RMSE':
            m = np.sqrt(np.mean(
                (metrics_dict['mean'] - metrics_dict['sim'][np.newaxis, ...]) ** 2,
                axis=0))
        elif metric[0] == 'bias':
            m = np.mean(metrics_dict['mean'], axis=0) - metrics_dict['sim']
        elif metric[0] == 'MAD':
            m = np.mean(
                np.abs(metrics_dict['mean'] - metrics_dict['sim'][np.newaxis, ...]),
                axis=0)
        elif metric[0] == 'variance':
            m = np.mean(metrics_dict['variance'], axis=0)
        elif metric[0] == 'coverage':
            def _coverage(target):
                covbool = np.abs(metrics_dict['ensemble_quantile'] - 0.5) < target / 2
                return np.mean(covbool, axis=0)
            m = np.stack([_coverage(target) for target in metric[1]], axis=-1)
        elif metric[0] == 'quantile':
            m = np.nanmean(metrics_dict['quantile'], axis=0)
        else:
            raise NotImplementedError(f'{metric} not supported.')
        return m

    @property
    def _default_validation_metrics(self):
        return [('RMSE',), ('bias',), ('MAD',), ('variance',), ('coverage', [0.5, 0.8]),
                ('quantile', (0.1, 0.9))]

    def _validation_metrics(self, metrics_dict, metrics=None):
        if metrics is None:
            metrics = self._default_validation_metrics
        res = {'meta_validation':{}}
        for metric in metrics:
            res[metric[0]] = self._validation_metric(metrics_dict, metric)
            res['meta_validation'][metric[0]] = metric[1:]
        return res

    def _replicates(self, prior=False):
        return [0] if prior else None

    def _assemble_metrics(
            self, pathout, param='e', metrics=None, delete_temp=False, indranges=None,
            prior=False):
        res = {}
        meta = {}
        suffix = self._suffix(param, indranges, prior)
        for r in self._replicate_generator(pathout, replicates=self._replicates(prior)):
            fnr = self.filename_metrics(pathout, r, suffix=suffix)
            res_r = load_object(fnr)
            for metric in res_r:
                if metric not in res:
                    res[metric] = [res_r[metric][0]]
                    meta[metric] = res_r[metric][1]
                else:
                    res[metric].append(res_r[metric][0])
        for metric in res:
            res[metric] = np.concatenate(res[metric], axis=0)
        res['meta'] = meta
        res['ygrid'] = self.ygrid
        if indranges is not None:
            res['indranges'] = indranges
            res['sim'] = self.prescribed_mean_period(indranges, param=param)
        else:
            res['sim'] = self.prescribed(param=param)

        res_metrics = self._validation_metrics(res, metrics=metrics)
        res.update(res_metrics)
        save_object(res, self.filename_metrics(pathout, None, suffix=suffix))
        if delete_temp:
            for r in self._replicate_generator(pathout, replicates=None):
                try:
                    self.filename_metrics(pathout, r, suffix=suffix).unlink()
                except:
                    pass

class InversionSimulatorIS(InversionSimulator):

    def _logweights_single(self, sim_obs_jsim, pred_scenes):
        from inference import lw_mvnormal, psislw
        from warnings import warn
        warn("Deprecated non-vectorized importance sampling")
        lw = lw_mvnormal(
            sim_obs_jsim[np.newaxis, ...], self.C_obs[np.newaxis, ...], pred_scenes)
        lw_ps, _ = psislw(lw)
        return lw_ps

    def _logweights(self, sim_obs, pred_scenes):
        from inference import lw_mvnormal, psislw
        _C_obs = np.broadcast_to(self.C_obs, (sim_obs.shape[0],) + self.C_obs.shape)
        lw = lw_mvnormal(
            sim_obs, _C_obs, pred_scenes)
        lw_ps, _ = psislw(lw)
        return lw_ps

    def logweights(self, replicates=10, pathout=None, n_jobs=-1):
        if pathout is None:
            pathout = Path.cwd()
            import warnings
            warnings.warn(f'Storing data in {pathout}')
        pred_scenes = self.predictions_scenes
        child_states = self.rng.bit_generator._seed_seq.spawn(replicates)
        def _export(r):
            sim_obs_r = self.simulated_observations(rng=child_states[r])
            lw_r = self._logweights(sim_obs_r, pred_scenes)
            save_object(lw_r, self.filename_sim(pathout, r))
            save_object(sim_obs_r, self.filename_simobs(pathout, r))
        from joblib import Parallel, delayed
        Parallel(n_jobs=n_jobs)(delayed(_export)(r) for r in range(replicates))

    def results(self, pathout, prior=False, replicates=None):
        lw_list = []
        simobs = []
        for r in self._replicate_generator(pathout, replicates):
            lw_r = load_object(self.filename_sim(pathout, r))
            lw_list.append(lw_r)
            simobs.append(load_object(self.filename_simobs(pathout, r)))
        lw, simobs = np.array(lw_list), np.array(simobs)
        if prior:
            lw = np.ones(lw.shape, dtype=np.float64)
        return SimInvEnsembleIS(self, lw, simobs=simobs)

    def inference(self, replicates=10, pathout=None, n_jobs=-1):
        return self.logweights(replicates=replicates, pathout=pathout, n_jobs=n_jobs)

class InversionSimulatorGM(InversionSimulator):

    def __init__(
            self, predens=None, predens_sim=None, K=3, rng=None, method_condition='unobserved'):
        # K: number of mixtures
        super().__init__(predens=predens, predens_sim=predens_sim, rng=rng)
        self.K = K
        self.method_condition = method_condition
    
    def register_variables(self, variables=None):
        if variables is None:
            variables = (('yf', {'ind_scene': [self.ind_scenes[-1]]}),
                         ('e', {'indranges': [(self.ind_scenes[-4], self.ind_scenes[-1])]}))
        self.variables = variables
        
    def _posterior(self, sim_obs, pred_scenes, unobserved):
        from inference import GaussianMixtureDistribution
        samples = np.concatenate((unobserved, pred_scenes), axis=-1)
        gmj = GaussianMixtureDistribution.from_samples(samples, self.K)
        _C_obs = np.broadcast_to(self.C_obs, (sim_obs.shape[0],) + self.C_obs.shape)
        gmp = gmj.conditional(sim_obs, C_obs=_C_obs, method_condition=self.method_condition)
        return gmp.to_array()

    @property
    def predicted_variables(self):
        unobserved_list = [self._predicted_variable(*v) for v in self.variables]
        unobserved = np.concatenate(unobserved_list, axis=-1)
        return unobserved

    def _predicted_variable(self, param, param_dict={}):
        # reverse engineering for gaussian mixture workflow
        if 'indranges' in param_dict:
            assert 'ind' not in param_dict
            self.predens.predict_mean_period(param_dict['indranges'], param=param)
            p = self.predens.results[f'{param}_mean_period']
        else:
            p = self.predens.results[param]
            if 'ind_scene' in param_dict:
                p = p[:, param_dict['ind_scene']]
        return p

    @property
    def _default_validation_metrics(self):
        return [('RMSE',), ('bias',), ('MAD',), ('variance',)]

    def posterior(self, replicates=10, pathout=None, n_jobs=-1):
        if pathout is None:
            pathout = Path.cwd()
            import warnings
            warnings.warn(f'Storing data in {pathout}')
        pred_scenes = self.predictions_scenes  # predicted observations
        unobserved = self.predicted_variables  # needed for GM
        child_states = self.rng.bit_generator._seed_seq.spawn(replicates)
        def _export(r):
            sim_obs_r = self.simulated_observations(rng=child_states[r])
            p_r = self._posterior(sim_obs_r, pred_scenes, unobserved)
            save_object(p_r, self.filename_sim(pathout, r))
            save_object(sim_obs_r, self.filename_simobs(pathout, r))
        from joblib import Parallel, delayed
        Parallel(n_jobs=n_jobs)(delayed(_export)(r) for r in range(replicates))

    def results(self, pathout, prior=False, replicates=None):
        gm_array_list = []
        simobs = []
        for r in self._replicate_generator(pathout, replicates):
            gm_array_r = load_object(self.filename_sim(pathout, r))
            gm_array_list.append(gm_array_r)
            simobs.append(load_object(self.filename_simobs(pathout, r)))
        gm_arr, simobs = np.stack(gm_array_list, axis=1), np.array(simobs) # axis 0 needs to be components
        if prior:
            raise NotImplementedError("Prior not implemented for Gaussian Mixtures")
        return SimInvEnsembleGM(self, gm_arr, simobs=simobs)

    def inference(self, replicates=10, pathout=None, n_jobs=-1):
        return self.posterior(replicates=replicates, pathout=pathout, n_jobs=n_jobs)

class SimInvEnsemble():

    @abstractmethod
    def __init__(self, invsim, infres, simobs=None):
        pass

    def predictions(self, param='e', p=None):
        if p is None:
            p = self.invsim.predens.results[param]
        return p

    def prescribed(self, param='e'):
            return self.invsim.predens_sim.results[param]

    def observed(self, replicate=None):
        if replicate is None:
            return self.simobs
        else:
            return self.simobs[replicate, ...]

    def predicted_mean_period(self, indranges, param='e'):
        return self.invsim.predens._mean_period(self.invsim.predens.results, indranges, param=param)

    def prescribed_mean_period(self, indranges, param='e'):
        return self.invsim.predens_sim._mean_period(self.invsim.predens_sim.results, indranges, param=param)

    @property
    def depth(self):
        return self.invsim.predens.results['depth']

    @property
    def dy(self):
        return self.invsim.predens.results['dy']

    @property
    def ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

    @property
    def ind_scenes(self):
        return self.invsim.ind_scenes

    def export_metrics(self, fnout, param='e', metrics=None, indranges=None):
        results = {}
        if metrics is None:
            metrics = [
                ('mean',), ('variance',), ('ensemble_quantile',), ('quantile', (0.1, 0.9))]
        for metric in metrics:
            mr = self._metric(
                metric[0], param=param, metric_args=metric[1:], indranges=indranges)
            results[metric[0]] = (mr, metric[1:])
        save_object(results, fnout)

    def _metric(self, metric, param='e', metric_args=(), indranges=None):
        if indranges is None:
            p = self.predictions(param=param)
            ref = self.prescribed(param)
        else:
            p = self.predicted_mean_period(indranges, param=param)
            ref = self.prescribed_mean_period(indranges, param=param)
        if metric == 'mean':
            return self.mean(p=p)
        elif metric == 'variance':
            return self.variance(p=p)
        elif metric == 'quantile':
            return self.quantile(metric_args[0], p=p)
        elif metric == 'ensemble_quantile':
            return self._ensemble_quantile(p, ref)
        else:
            raise NotImplementedError(f'{metric} not known')

    @abstractmethod
    def mean(self, param='e', p=None, replicate=None):
        pass

    @abstractmethod
    def variance(self, param='e', p=None, replicate=None):
        pass

class SimInvEnsembleIS(SimInvEnsemble):
    def __init__(self, invsim, infres, simobs=None):
        self.lw = infres
        self.invsim = invsim
        self.simobs = simobs

    def moment(self, param='e', replicate=None, power=1, p=None):
        from inference import expectation
        p = self.predictions(param=param, p=p)
        if len(p.shape) == 1:
            p = p[:, np.newaxis]
            shrink = True
        else:
            shrink = False
        if replicate is None:
            lw_ = self.lw
        else:
            lw_ = self.lw[replicate, ...]
        postmean = expectation(
            np.power(p, power), lw_, normalize=True)
        if shrink:
            postmean = postmean[..., 0]
        return postmean

    def variance(self, param='e', replicate=None, p=None):
        # improvement needed to deal with numerical issues
        p = self.predictions(param=param, p=p)
        var = (self.moment(param=None, replicate=replicate, p=p, power=2)
               -self.moment(param=None, replicate=replicate, p=p, power=1) ** 2)
        return var

    def mean(self, param='e', p=None, replicate=None):
        return self.moment(param=param, p=p, replicate=replicate)

    def _ensemble_quantile(self, p, ref):
        # computes quantile of "truth" with respect to ensemble distribution
        from inference import ensemble_quantile as eq
        return eq(p, self.lw, ref)

    def quantile(
            self, quantiles, param='e', replicate=None, jsim=None, smooth=None, steps=8,
            p=None, method='bisection'):
        from inference import quantile as quant
        repl_slice = np.s_[:] if replicate is None else np.s_[replicate]
        jsim_slice = np.s_[:] if jsim is None else np.s_[jsim]
        lw_ = self.lw[(repl_slice, jsim_slice, Ellipsis)]
        if param is not None and p is None:
            p = self.predictions(param)
        postquant = quant(
            p, lw_, quantiles, method=method, steps=steps,
            normalize=True, smooth=smooth)
        return postquant

    def frac_thawed(self, jsim=0, replicate=0):
        yf = self.predictions('yf')[..., self.ind_scenes[-1]]
        from inference.isi import sumlogs
        lw_ = self.lw[replicate, jsim, ...]
        lw_ -= sumlogs(lw_)[np.newaxis]
        w_ = np.exp(lw_)[np.newaxis,:] * np.ones((self.ygrid.shape[0], lw_.shape[0]))
        np.putmask(w_, self.ygrid[:, np.newaxis] > yf[np.newaxis,:], 0)
        frac_thawed = np.sum(w_, axis=1)
        return frac_thawed

class SimInvEnsembleGM(SimInvEnsemble):
    def __init__(self, invsim, invres, simobs=None):
        self.invsim = invsim
        self.simobs = simobs
        from inference import GaussianMixtureDistribution
        self.gmd = GaussianMixtureDistribution.from_array(invres)
    
    @property
    def variables(self):
        return self.invsim.variables
    
    def _indices_variables(self, param='e', indranges=None):
        if len(self.variables) > 1: 
            raise NotImplementedError("variable_indices function needs to be generalized")
        variable = self.variables[0]
        assert param == variable[0]
        if indranges is not None:
            indices = np.array([variable[1]['indranges'].index(indrange) for indrange in indranges])
        else:        
            indices = [0]
        return indices        
    
    def _metric(self, metric, param='e', metric_args=(), indranges=None):
        indices = self._indices_variables(param=param, indranges=indranges)
        if metric == 'mean':
            return self.mean(indices=indices)
        elif metric == 'variance':
            return self.variance(indices=indices)
        else:
            raise NotImplementedError(f'{metric} not known')
                                          
    def variance(self, indices=None, p=None, replicate=None):
        if replicate is not None: raise NotImplementedError()
        var = self.gmd.variance(indices=indices)
        return var

    def mean(self, indices=None, p=None, replicate=None):
        if replicate is not None: raise NotImplementedError()
        _mean = self.gmd.mean(indices=indices)
        return _mean

