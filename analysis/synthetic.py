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

    def predict(self, forcing, n_jobs=4, **kwargs):
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

    def logweights(self, replicates=10, pathout=None, n_jobs=8):
        if pathout is None:
            pathout = os.getcwd()
            import warnings
            warnings.warn(f'Storing data in {pathout}')
        pred_scenes = self.predictions_scenes
        child_states = self.rng.bit_generator._seed_seq.spawn(replicates)
        def _export(r):
            sim_obs_r = self.simulated_observations(rng=child_states[r])
            lw_r = self._logweights(sim_obs_r, pred_scenes)
            save_object(lw_r, os.path.join(pathout, f'sim_{r}.npy'))
            save_object(sim_obs_r, os.path.join(pathout, f'simobs_{r}.npy'))
        from joblib import Parallel, delayed
        Parallel(n_jobs=n_jobs)(delayed(_export)(r) for r in range(replicates))

    def export(self, fnout):
        isdict = {
            'predens': self.predens, 'predens_sim': self.predens_sim, 'rng': self.rng,
            'ind_scenes': self.ind_scenes, 'C_obs': self.C_obs}
        save_object(isdict, fnout)

    def results(self, pathout, replicates=None):
        r = 0
        lw = []
        simobs = []
        while replicates is None or r < replicates :
            fn_r = os.path.join(pathout, f'sim_{r}.npy')
            if os.path.exists(fn_r):
                lw_r = load_object(fn_r)
                lw.append(lw_r)
                simobs.append(load_object(os.path.join(pathout, f'simobs_{r}.npy')))
                r += 1
            else:
                break
        return simInvEnsemble(self, np.array(lw), simobs=np.array(simobs))

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

    def quantile(self, quantiles, param='e', replicate=0, jsim=0, smooth=None):
        from inference import quantile as quant
        repl_slice = np.s_[:] if replicate is None else np.s_[replicate]
        jsim_slice = np.s_[:] if jsim is None else np.s_[jsim]
        lw_ = self.lw[(repl_slice, jsim_slice, Ellipsis)]
#         lw_ = self.lw[replicate, jsim, ...][np.newaxis, :]
        postquant = quant(
            self.invsim.predens.results[param], lw_, quantiles, normalize=True, 
            smooth=smooth)
        return postquant

    def prescribed(self, param='e'):
        return self.invsim.predens_sim.results[param]

    def observed(self, replicate=None):
        if replicate is None:
            return self.simobs
        else:
            return self.simobs[replicate, ...]

    @property
    def ygrid(self):
        return np.arange(0, self.depth, step=self.dy)

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
#         jyrange = []
#         for ind in indrange:
#             yf = self.invsim.predens.results['yf'][:, ind]
#             jyrange.append(
#                 np.argmin(np.abs(yf[:, np.newaxis] -  ygrid[np.newaxis, :]), axis=1))
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
        smooth_quantile = 2
        ygrid = self.ygrid
        e_inv = self.moment('e', replicate=replicate)
        e_inv_std = np.sqrt(self.variance('e', replicate=replicate))
        if show_quantile:
            e_inv_q = self.quantile(
                [0.1, 0.5, 0.9], 'e', replicate=replicate, jsim=jsim, 
                smooth=smooth_quantile)
        e_sim = self.prescribed('e')
        s_sim = self.prescribed('s_los')
        s_obs = self.observed(replicate=replicate)
        s_pred = self.moment('s_los', replicate=replicate)
        fig, axs = plt.subplots(ncols=2, sharey=False)
        fig.set_size_inches((8, 3), forward=True)
        days = np.arange(s_sim.shape[1])
        axs[0].plot(
            days[self.invsim.ind_scenes[1:]], s_obs[jsim, ...], lw=0.0, c='k', alpha=0.6,
            marker='o', mfc='k', mec='none', ms=4)
        axs[0].plot(
            days, s_pred[jsim, ...] - s_pred[jsim, self.invsim.ind_scenes[0]],
            c='#aa9966', lw=1.0)
        axs[0].plot(
            days, s_sim[jsim, ...] - s_sim[jsim, self.invsim.ind_scenes[0]],
            lw=1.0, c='#000000')
        axs[1].plot(e_sim[jsim, :], ygrid, lw=1.0, c='#000000')
        alpha = self.frac_thawed(replicate=replicate, jsim=jsim)
        for jdepth in np.arange(ygrid.shape[0] - 1):
            axs[1].plot(
                e_inv[jsim, jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=1.0,
                c='#aa9966', alpha=alpha[jdepth])
            if show_quantile:
                axs[1].plot(
                    e_inv_q[1][jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=0.5,
                    c='#aa9966', alpha=alpha[jdepth], ls='--')
                axs[1].plot(
                    e_inv_q[0][jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=0.5,
                    c='#9999ee', alpha=alpha[jdepth])
                axs[1].plot(
                    e_inv_q[2][jdepth:jdepth + 2], ygrid[jdepth:jdepth + 2], lw=0.5,
                    c='#9999ee', alpha=alpha[jdepth])
            else:
                axs[1].plot(
                    e_inv[jsim, jdepth:jdepth + 2] + e_inv_std[jsim, jdepth:jdepth + 2],
                     ygrid[jdepth:jdepth + 2], lw=0.5, c='#9999ee', alpha=alpha[jdepth])
                axs[1].plot(
                    e_inv[jsim, jdepth:jdepth + 2] - e_inv_std[jsim, jdepth:jdepth + 2],
                    ygrid[jdepth:jdepth + 2], lw=0.5, c='#9999ee', alpha=alpha[jdepth])
        if ymax is None: ymax = ygrid[-1]
        axs[1].set_ylim((ymax, ygrid[0]))
        plt.show()

    @property
    def depth(self):
        return self.invsim.predens.results['depth']

    @property
    def dy(self):
        return self.invsim.predens.results['dy']