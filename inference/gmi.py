'''
Created on Feb 29, 2020

@author: simon
'''
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_normal
from inference.isi import _sqr_eigen, _nondata_terms_mvnormal, invert_nonzero, sumlogs

def _sqr(A, cond_thresh=1e-6):
    lam_sqr, Q, ind = _sqr_eigen(A, cond_thresh=cond_thresh)
    sqr = Q * lam_sqr[:, np.newaxis,:]
    lam_sqr[ind['singular']] = 1
    lam_isqr = lam_sqr ** (-1)
    lam_isqr[ind['singular']] = 0
    isqrT = Q * lam_isqr[:, np.newaxis,:]
    return sqr, isqrT, ind, lam_sqr

class GaussianMixtureDistribution():
    def __init__(self, means, covariances, weights, precisions_cholesky=None):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self._precisions_cholesky = precisions_cholesky
    
    @property
    def K(self):
        return self.means.shape[0]
    
    # for interoperability with sklearn
    @property
    def means_(self):
        return self.means
    
    @property
    def covariances_(self):
        return self.covariances
    
    @property
    def weights_(self):
        return self.weights
    
    @property
    def precisions_cholesky_(self):
        if self._precisions_cholesky is not None:
            return self._precisions_cholesky
        else:
            raise NotImplementedError("Computation of cholesky f. of precision matrices not implemented")
    
    @classmethod
    def from_samples(cls, samples, K=2, random_state=None, n_init=8):
        from sklearn.mixture import GaussianMixture
        if random_state is None: random_state = 999
        gm = GaussianMixture(
            random_state=random_state, n_components=K, covariance_type='full', n_init=n_init,
            init_params='random', verbose=0).fit(samples)
        if not gm.converged_: raise ValueError("Gaussian Mixture fitting did not converge")
        return cls(gm.means_, gm.covariances_, gm.weights_, precisions_cholesky=gm.precisions_cholesky_)
    
    @property
    def _covariances_vectorized(self):
        P = self.covariances.shape[-1]
        ind = np.tril_indices(P)
        ind_ = (slice(None),) * (len(self.covariances.shape) - 2) + ind
        return self.covariances[ind_]        
        
    def to_array(self):
        arr = np.concatenate(
            (self.means, self._covariances_vectorized, self.weights[..., np.newaxis]), axis=-1)
        return arr
    
    @classmethod
    def from_array(cls, arr):
        S = arr.shape[-1]
        P = int(-3/2 + np.sqrt(9/4 + 2 * S - 2))
        assert P + (P * (P+1)) // 2 + 1 == S
        means = arr[..., :P]
        weights = arr[..., -1]
        cov = np.zeros(means.shape + (P, ), dtype=means.dtype)
        ind = np.tril_indices(P)
        cov[(slice(None),) * (len(means.shape) - 1) + ind] = arr[..., P:-1]
        cov[(slice(None),) * (len(means.shape) - 1) + (np.arange(P, dtype=np.int64),)*2] = 0
        cov[(slice(None),) * (len(means.shape) - 1) + (ind[1], ind[0])] += arr[..., P:-1]
        return cls(means, cov, weights)
        
    def conditional(self, y_obs, C_obs=None, cond_thresh=1e-6, method_condition='unobserved'):
        # conditions on the last P variables, yielding new gaussian mixture model
        # y_obs: last P variables are assumed to be observed directly, disturbed by noise with cov. C_obs
        P = y_obs.shape[1]  # number of observations
        M = y_obs.shape[0]  # number of replicates
        Q = self.means_.shape[1]  # dimension of prior RV considered
        Qout = Q if not 'unobserved' in method_condition else Q - P # number of output variables

        if C_obs is None: 
            if 'sqr' in method_condition: 
                raise ValueError("Square root method requires positive definite C_obs")
            C_obs = 0
        else:
            assert C_obs.shape == (M, P, P)
        if 'sqr' in method_condition:
            V, _, ind_V, _ = _sqr(C_obs, cond_thresh=cond_thresh)
        else:
            V, ind_V = None, None
            
        logpi_p = np.empty((self.K, M))
        mu_p = np.empty((self.K, M, Qout))
        Sigma_p = np.empty((self.K, M, Qout, Qout))
    
        for k in range(self.K):
            mu_p[k, ...], Sigma_p[k, ...], logpi_p[k, ...] = self._condition_component(
                k, y_obs, C_obs, V=V, ind_V=ind_V, cond_thresh=cond_thresh,
                method_condition=method_condition)
    
        # clean up
        ind_invalid = np.any(np.isnan(logpi_p), axis=0)
        logpi_p[:, ind_invalid] = 0.0
        # normalize weights
        pi_p = np.exp(logpi_p - sumlogs(logpi_p, axis=0)[np.newaxis, ...])
        mu_p[:, ind_invalid] = np.nan
        Sigma_p[:, ind_invalid] = np.nan
        pi_p[:, ind_invalid] = np.nan
        return GaussianMixtureDistribution(mu_p, Sigma_p, pi_p)

    def _condition_component(
            self, k, y_obs, C_obs, V=None, ind_V=None, cond_thresh=1e-6, method_condition='full'):
        # conditions on component k (posterior mean and variance) + new weight
        P = y_obs.shape[1]  # number of observations
        Q = self.means_.shape[1]  # dimension of prior RV considered
        q_y = Q - P
        mu_k, Sigma_k, pi_k = self.means[k, ...], self.covariances[k, ...], self.weights[k]
        if method_condition in ('sqr', 'sqr_unobserved'):
            # square root implementation following Angus Andrews:
            # A Square Root Formulation of the Kalman Covariance Equations
            Wp_k = solve_triangular(self.precisions_cholesky_[k, ...], np.eye(Q)).T
            Z_k = np.transpose(Wp_k)[:, q_y:]
            U_k, U_invT_k, ind_U_k, U_k_lam_sqr = _sqr(
                Sigma_k[q_y:, q_y:] + C_obs, cond_thresh=cond_thresh)
            y_k_prior = mu_k[q_y:]
            if V is None and ind_V is None:
                V, _, ind_V, _ = _sqr(C_obs, cond_thresh=cond_thresh)        
            UpV_k_inv = np.linalg.pinv(U_k + V, rcond=cond_thresh)
            UPD_k = np.eye(Q)[np.newaxis, ...] - np.einsum(
                'qb, mbc, mcd, ed -> mqe', Z_k, U_invT_k, UpV_k_inv, Z_k, optimize=True)
            # W_k neither triangular nor symmetric
            W_k = np.einsum('qb, mbc -> mqc', Wp_k, UPD_k, optimize=True)
            Sigma_p_k = np.einsum('mqb, mcb -> mqc', W_k, W_k, optimize=True)
            mu_p_k = mu_k[np.newaxis,:] + np.einsum(
                'pa, ab, mbc, mdc, md -> mp', Wp_k, Z_k, U_invT_k, U_invT_k, y_obs - y_k_prior,
                optimize=True)
            logdetfac, normfac = _nondata_terms_mvnormal(
                invert_nonzero(U_k_lam_sqr, ind_zero=ind_U_k['singular']),
                ind_singular=ind_U_k['singular'])
            prod = np.einsum('mqp, mq -> mp', U_invT_k, y_obs - y_k_prior)
            maha = -0.5 * np.sum(prod ** 2, axis=1)
            logpi_p_k = np.log(pi_k) + maha + logdetfac + normfac
            if method_condition == 'sqr_unobserved':
                sl = slice(None, q_y)
                mu_p_k, Sigma_p_k = mu_p_k[..., sl], Sigma_p_k[..., sl, sl]
            mu_p_k[ind_V['invalid'], ...] = np.nan
            Sigma_p_k[ind_V['invalid'], ...] = np.nan
            logpi_p_k[ind_V['invalid'], ...] = np.nan                        
        elif method_condition in ('full', 'unobserved'):
            y_k_prior = mu_k[q_y:]
            # only compute for indices up to q_y if unobserved
            sl = slice(None, None) if method_condition == 'full' else slice(None, q_y)
            Sigma_k_off = Sigma_k[sl, q_y:]
            # adding observation covariance (works if observation noise is Gaussian)
            Sigma_k_obs = Sigma_k[q_y:, q_y:] + C_obs
            Sigma_k_obs_inv = np.linalg.pinv(Sigma_k_obs, rcond=cond_thresh)
            # a few shortcuts
            dy = y_obs - y_k_prior
            _mu_k = mu_k[np.newaxis, sl]
            _Sigma_k = Sigma_k[np.newaxis, sl, sl]
            # update mu and Sigma using standard mv normal 
            mu_p_k = _mu_k + np.einsum('pb, mbc, mc -> mp', Sigma_k_off, Sigma_k_obs_inv, dy, optimize=True)
            Sigma_p_k = _Sigma_k - np.einsum(
                'pb, mbc, dc -> mpd', Sigma_k_off, Sigma_k_obs_inv, Sigma_k_off, optimize=True)
            # compute new weight for Gaussian mixture (not normalzied yet)
            log_p_y_obs = -0.5 * (
                P * np.log(2 * np.pi) + np.linalg.slogdet(Sigma_k_obs)[1]
                +np.einsum('mc, md, mcd -> m', dy, dy, Sigma_k_obs_inv))
            logpi_p_k = np.log(pi_k) + log_p_y_obs
        else:
            raise NotImplementedError
        return mu_p_k, Sigma_p_k, logpi_p_k        


    def mean(self, indices=None):
        # marginal mean (just weighted mean)
        _mu = self.means if indices is None else self.means[..., indices]
        mean = np.einsum('i...k, i... -> ...k', _mu, self.weights)
        return mean
        
    def covariance(self, indices=None):
        # marginal covariance matrix
        mean = self.mean(indices=indices)
        mus = self.means if indices is None else self.means[..., indices]
        mu_dev = mus - mean[np.newaxis, ...]
        pis =  self.weights
        Sigs = self.covariances[..., indices, :][..., indices] if indices is not None else self.covariances
        # from Mode-finding for mixtures of Gaussian distributions
        Sigma = np.einsum(
            'i...kl, i... -> ...kl', Sigs + mu_dev[..., np.newaxis] * mu_dev[..., np.newaxis,:], pis)
        return Sigma

    def variance(self, indices=None):
        Sigma = self.covariance(indices=indices)
        return np.diagonal(Sigma, axis1=-2, axis2=-1)


if __name__ == '__main__':
    N = 2048  # samples
    P = 8
    H = np.linspace(1, 4, num=P)[np.newaxis,:]
    rs = np.random.RandomState(seed=5)
    parms = rs.uniform(low=0.0, high=0.8, size=(N, 1))
    y_ref = parms * H
    y_ref += rs.normal(scale=0.3 , size=(N, P))

    # fit
    samples = np.concatenate((parms, y_ref), axis=-1)
    # gm = fit_gaussian_mixture(samples,k=2)
    gmj = GaussianMixtureDistribution.from_samples(samples, K=2) 
    # observations
    M = 512
    parm0 = np.array([0.7])
    sigma_obs = 0.10
    y_obs = rs.normal(scale=sigma_obs, size=(M, P))
    y_obs[...] += parm0[np.newaxis,:] * H
    C_obs = np.zeros((M, P, P))
    C_obs[:, ...] = sigma_obs ** 2 * np.eye(P)[np.newaxis, ...]
    # mu_p, Sigma_p, pi_p = posterior_gm_mvnormal(y_obs, C_obs, gm, method_condition='sqr')
    # mu_pu, Sigma_pu, pi_pu = posterior_gm_mvnormal(y_obs, C_obs, gm, method_condition='unobserved')
    gmp = gmj.conditional(y_obs, C_obs, method_condition='sqr_unobserved')
    gmp2 = gmj.conditional(y_obs, C_obs, method_condition='unobserved')
    print(gmp2.means_.shape)

    
    