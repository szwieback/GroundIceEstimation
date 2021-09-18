'''
Created on Aug 11, 2021

@author: simon
'''

from simulation.stefan import stefan_integral_balance
from analysis.ioput import save_object, load_object

import numpy as np
import os

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

    def predict(self, forcing, n_jobs=-2, **kwargs):
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
            assert C_obs.shape[0] == s.shape[1]
            obs_noise = rng.multivariate_normal(
                np.zeros(s.shape[1]), C_obs, size=(s.shape[0],))
            s += obs_noise
        return s

class inversionSimulator():
    # hard-coded to Gaussian likelihood with IS
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
        invsim.register_observations(isdict['ind_scenes'], isdict['C_obs'])
        return invsim

    def register_observations(self, ind_scenes, C_obs):
        self.ind_scenes = ind_scenes
        self.C_obs = C_obs

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
        return self.predens.results['depth']

    @property
    def dy(self):
        return self.predens.results['dy']

    @property
    def ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

    def simulated_observations(self, rng=None):
        rng_ = self.rng if rng is None else np.random.default_rng(rng)
        return self.predens_sim.extract_predictions(
            self.ind_scenes, C_obs=self.C_obs, rng=rng_)

    def _logweights_single(self, sim_obs_jsim, pred_scenes, jsim=0):
        from inference import lw_mvnormal, psislw
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

    def logweights(self, replicates=10, pathout=None, n_jobs=-2):
        if pathout is None:
            pathout = os.getcwd()
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

    def export(self, fnout):
        isdict = {
            'predens': self.predens, 'predens_sim': self.predens_sim, 'rng': self.rng,
            'ind_scenes': self.ind_scenes, 'C_obs': self.C_obs}
        save_object(isdict, fnout)

    def _replicate_generator(self, pathout, replicates=None):
        if replicates is not None:
            for r in replicates:
                yield r
        else:
            from itertools import count
            for r in count(0):
                if os.path.exists(self.filename_sim(pathout, r)):
                    yield r
                else:
                    break
    def prescribed(self, param='e'):
        return self.predens_sim.results[param]

    def filename_sim(self, pathout, r):
        return os.path.join(pathout, f'sim_{r}.npy')

    def filename_simobs(self, pathout, r):
        return os.path.join(pathout, f'simobs_{r}.npy')

    def filename_metrics(self, pathout, r, suffix=None):
        if suffix is None:
            suffix_ = '_' + suffix
        else:
            suffix_ = ''
        return os.path.join(pathout, f'metrics{suffix_}_{r}.p')

    def results(self, pathout, replicates=None):
        lw = []
        simobs = []
        for r in self._replicate_generator(pathout, replicates=replicates):
            lw_r = load_object(self.filename_sim(pathout, r))
            lw.append(lw_r)
            simobs.append(load_object(self.filename_simobs(pathout, r)))
        return simInvEnsemble(self, np.array(lw), simobs=np.array(simobs))

    def export_metrics(self, pathout, param='e', metrics_ind=None, metrics=None, n_jobs=-2):
        def _export(r):
            sie = self.results(pathout, replicates=(r,))
            fnout = self.filename_metrics(pathout, r, suffix=param)
            sie.export_metrics(fnout, param=param, metrics=metrics_ind)
        from joblib import Parallel, delayed
        rgen = self._replicate_generator(pathout, replicates=None)
        Parallel(n_jobs=n_jobs)(delayed(_export)(r) for r in rgen)
        self._assemble_metrics(pathout, param='e', metrics=metrics, delete_temp=True)

    def _validation_metric(self, metrics_dict, metric):
        if metric[0] == 'RMSE':
            m = np.sqrt(np.mean(
                (metrics_dict['mean'] - metrics_dict['sim'][np.newaxis, ...])**2, 
                axis=0))
        elif metric[0] == 'bias':
            m = metrics_dict['mean'] - np.mean(metrics_dict['sim'], axis=0)
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
        return m

    def _validation_metrics(self, metrics_dict, metrics=None):
        if metrics is None:
            metrics = [
                ('RMSE',), ('bias',), ('MAD',), ('variance',), ('coverage', [0.5, 0.8])]
        res = {'meta_validation':{}}
        for metric in metrics:
            print(metric)
            res[metric[0]] = self._validation_metric(metrics_dict, metric)
            res['meta_validation'][metric[0]] = metric[1:]
        return res
    
    def _assemble_metrics(self, pathout, param='e', metrics=None, delete_temp=False):
        res = {}
        meta = {}
        for r in self._replicate_generator(pathout, replicates=None):
            fnr = self.filename_metrics(pathout, r, suffix=param)
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
        res['sim'] = self.prescribed(param=param)
        res_metrics = self._validation_metrics(res, metrics=metrics)
        res.update(res_metrics)
        save_object(res, os.path.join(pathout, 'metrics.p'))
        if delete_temp:
            for r in self._replicate_generator(pathout, replicates=None):
                try:
                    os.remove(self.filename_metrics(pathout, r, suffix=param))
                except:
                    pass

class simInvEnsemble():

    def __init__(self, invsim, lw, simobs=None):
        self.lw = lw
        self.invsim = invsim
        self.simobs = simobs

    def moment(self, param='e', replicate=None, power=1, p=None):
        from inference import expectation
        if param is not None:
            p = self.invsim.predens.results[param]
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
        if param is not None:
            p = self.invsim.predens.results[param]
        var = (self.moment(param=None, replicate=replicate, p=p, power=2)
               -self.moment(param=None, replicate=replicate, p=p, power=1) ** 2)
        return var

    def quantile(
            self, quantiles, param='e', replicate=None, jsim=None, smooth=None, steps=8,
            method='bisection'):
        from inference import quantile as quant
        repl_slice = np.s_[:] if replicate is None else np.s_[replicate]
        jsim_slice = np.s_[:] if jsim is None else np.s_[jsim]
        lw_ = self.lw[(repl_slice, jsim_slice, Ellipsis)]
        postquant = quant(
            self.invsim.predens.results[param], lw_, quantiles, method=method, steps=steps,
            normalize=True, smooth=smooth)
        return postquant

    def prescribed(self, param='e'):
        return self.invsim.predens_sim.results[param]

    def observed(self, replicate=None):
        if replicate is None:
            return self.simobs
        else:
            return self.simobs[replicate, ...]

    def frac_thawed(self, jsim=0, replicate=0):
        yf = self.invsim.predens.results['yf'][..., self.invsim.ind_scenes[-1]]
        from inference.isi import sumlogs
        lw_ = self.lw[replicate, jsim, ...]
        lw_ -= sumlogs(lw_)[np.newaxis]
        w_ = np.exp(lw_)[np.newaxis, :] * np.ones((self.ygrid.shape[0], lw_.shape[0]))
        np.putmask(w_, self.ygrid[:, np.newaxis] > yf[np.newaxis, :], 0)
        frac_thawed = np.sum(w_, axis=1)
        return frac_thawed

    def mean_period(self, indrange, param='e'):
        ygrid = self.ygrid
        p = self.invsim.predens.results[param]
        yfrange = [self.invsim.predens.results['yf'][:, ind] for ind in indrange]
        invalid = np.logical_or(
            yfrange[0][:, np.newaxis] > ygrid[np.newaxis, :],
            yfrange[1][:, np.newaxis] < ygrid[np.newaxis, :])
        np.putmask(p, invalid, np.nan)
        p_mean = np.nanmean(p, axis=1)
        return p_mean

    def plot(self, jsim=0, replicate=0, ymax=None, show_quantile=False):
        import matplotlib.pyplot as plt
        globfigparams = {
            'fontsize':8, 'family':'serif', 'usetex': True,
            'preamble': r'\usepackage{amsmath} \usepackage{times} \usepackage{mathtools}',
            'column_inch':229.8775 / 72.27, 'markersize':24, 'markercolour':'#AA00AA',
            'fontcolour':'#666666', 'tickdirection':'out', 'linewidth': 0.5,
            'ticklength': 2.50, 'minorticklength': 1.1 }
        plt.rc(
            'font', **{'size':globfigparams['fontsize'], 'family':globfigparams['family']})
        plt.rcParams['text.usetex'] = globfigparams['usetex']
        plt.rcParams['text.latex.preamble'] = globfigparams['preamble']
        plt.rcParams['legend.fontsize'] = globfigparams['fontsize']
        plt.rcParams['font.size'] = globfigparams['fontsize']
        plt.rcParams['axes.linewidth'] = globfigparams['linewidth']
        plt.rcParams['axes.labelcolor'] = globfigparams['fontcolour']
        plt.rcParams['axes.edgecolor'] = globfigparams['fontcolour']
        plt.rcParams['xtick.color'] = globfigparams['fontcolour']
        plt.rcParams['xtick.direction'] = globfigparams['tickdirection']
        plt.rcParams['ytick.direction'] = globfigparams['tickdirection']
        plt.rcParams['ytick.color'] = globfigparams['fontcolour']
        plt.rcParams['xtick.major.width'] = globfigparams['linewidth']
        plt.rcParams['ytick.major.width'] = globfigparams['linewidth']
        plt.rcParams['xtick.minor.width'] = globfigparams['linewidth']
        plt.rcParams['ytick.minor.width'] = globfigparams['linewidth']
        plt.rcParams['ytick.major.size'] = globfigparams['ticklength']
        plt.rcParams['xtick.major.size'] = globfigparams['ticklength']
        plt.rcParams['ytick.minor.size'] = globfigparams['minorticklength']
        plt.rcParams['xtick.minor.size'] = globfigparams['minorticklength']
        plt.rcParams['text.color'] = globfigparams['fontcolour']
        cols = {'true': '#000000', 'est': '#aa9966', 'unc': '#9999ee'}
        smooth_quantile = 2
        ygrid = self.ygrid
        e_inv = self.moment('e', replicate=replicate)
        e_inv_std = np.sqrt(self.variance('e', replicate=replicate))
        if show_quantile:
            e_inv_q = self.quantile(
                [0.1, 0.9], 'e', replicate=replicate, jsim=jsim,
                smooth=smooth_quantile)
        e_sim = self.prescribed('e')
        s_sim = self.prescribed('s_los')
        s_obs = self.observed(replicate=replicate)
        s_pred = self.moment('s_los', replicate=replicate)
        fig, axs = plt.subplots(ncols=2, sharey=False)
        plt.subplots_adjust(top=0.92, left=0.10, right=0.98, bottom=0.15, wspace=0.30)
        fig.set_size_inches((5, 2.5), forward=True)
        days = np.arange(s_sim.shape[1])
        axs[0].plot(
            days[self.invsim.ind_scenes[1:]], s_obs[jsim, ...], lw=0.0, c='k', alpha=0.6,
            marker='o', mfc='k', mec='none', ms=4)
        axs[0].plot(
            days, s_pred[jsim, ...] - s_pred[jsim, self.invsim.ind_scenes[0]],
            c=cols['est'], lw=1.0)
        axs[0].plot(
            days, s_sim[jsim, ...] - s_sim[jsim, self.invsim.ind_scenes[0]],
            lw=1.0, c=cols['true'])
        axs[0].set_xlabel('time since snow melt [d]')
        axs[0].set_ylabel('subsidence [m]')
        axs[1].plot(e_sim[jsim, :], ygrid, lw=1.0, c=cols['true'])
        alpha = self.frac_thawed(replicate=replicate, jsim=jsim)
        for jdepth in np.arange(ygrid.shape[0] - 1):
            axs[1].plot(
                e_inv[jsim, jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=1.0,
                c=cols['est'], alpha=alpha[jdepth])
            if show_quantile:
                axs[1].plot(
                    e_inv_q[0][jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=0.5,
                    c=cols['unc'], alpha=alpha[jdepth])
                axs[1].plot(
                    e_inv_q[1][jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=0.5,
                    c=cols['unc'], alpha=alpha[jdepth])
            else:
                axs[1].plot(
                    e_inv[jsim, jdepth:jdepth + 2] + e_inv_std[jsim, jdepth:jdepth + 2],
                     ygrid[jdepth:jdepth + 2], lw=0.5, c=cols['unc'], alpha=alpha[jdepth])
                axs[1].plot(
                    e_inv[jsim, jdepth:jdepth + 2] - e_inv_std[jsim, jdepth:jdepth + 2],
                    ygrid[jdepth:jdepth + 2], lw=0.5, c=cols['unc'], alpha=alpha[jdepth])
        if ymax is None: ymax = ygrid[-1]
        axs[1].set_ylim((ymax, ygrid[0]))
        axs[1].set_ylabel('Depth [m]')
        axs[1].set_xlabel('Excess ice content [-]')
        titles = ['observations', 'ice content profile']
        for jax, ax in enumerate(axs):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.text(
                0.50, 1.05, titles[jax], c='k', transform=ax.transAxes,
                ha='center', va='baseline')
        plt.show()

    @property
    def depth(self):
        return self.invsim.predens.results['depth']

    @property
    def dy(self):
        return self.invsim.predens.results['dy']

    @property
    def ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

    def mean(self, param='e', replicate=None):
            return self.moment(param=param, replicate=replicate)

    def _metric(self, metric, param='e', replicate=None, metric_args=()):
        if metric == 'mean':
            return self.mean(param=param, replicate=replicate)
        elif metric == 'variance':
            return self.variance(param=param, replicate=replicate)
        elif metric == 'quantile':
            return self.quantile(metric_args[0], param=param, replicate=replicate)
        elif metric == 'ensemble_quantile':
            return self.ensemble_quantile(param=param, replicate=replicate)
        else:
            raise NotImplementedError(f'{metric} not known')

    def ensemble_quantile(self, param='e', replicate=None):
        # computes quantile of "truth" with respect to ensemble distribution
        if replicate is not None: raise NotImplementedError
        from inference import ensemble_quantile as eq
        ref = self.prescribed(param)
        p = self.invsim.predens.results[param]
        return eq(p, self.lw, ref)

    def export_metrics(self, fnout, param='e', metrics=None):
        results = {}
        if metrics is None:
            metrics = [('mean',), ('variance',), ('ensemble_quantile',)]
#             metrics = [('quantile', [0.10, 0.25, 0.50, 0.75, 0.90])]
        for metric in metrics:
            mr = self._metric(
                metric[0], param=param, replicate=None, metric_args=metric[1:])
            results[metric[0]] = (mr, metric[1:])
        save_object(results, fnout)

