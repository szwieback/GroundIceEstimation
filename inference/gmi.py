'''
Created on Feb 29, 2020

@author: simon
'''
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_normal

from inference.sampling import _sqr_eigen, _nondata_terms_mvnormal, invert_nonzero, sumlogs

def fit_gaussian_mixture(
        samples, weight_concentration_prior=1e-3, random_state=None,
        n_components=4, covariance_type='full', n_init=10, init_params='random',
        ):
    # uses variational Bayes
    bgm = BayesianGaussianMixture(
        weight_concentration_prior=weight_concentration_prior, random_state=random_state,
        n_components=n_components, covariance_type=covariance_type, n_init=n_init,
        init_params=init_params,).fit(samples)
    assert bgm.converged_
    return bgm

def _sqr(A, cond_thresh=1e-6,):
    lam_sqr, Q, ind = _sqr_eigen(A, cond_thresh=cond_thresh)
    sqr = Q * lam_sqr[:, np.newaxis, :]
    lam_sqr[ind['singular']] = 1
    lam_isqr = lam_sqr ** (-1)
    lam_isqr[ind['singular']] = 0
    isqrT = Q * lam_isqr[:, np.newaxis, :]
    return sqr, isqrT, ind, lam_sqr

def _condition_gm(
        y_obs, C_obs, gm, k=0, H=None, V=None, ind_V=None, cond_thresh=1e-6, 
        method_condition='square_root'):
    # method_condition: 'square_root' or 'full' 
    # (latter does not handle nonpos-def matrices, mainly as external check)
    if H is not None: 
        raise NotImplementedError
    mu = gm.means_
    Sigma = gm.covariances_
    pi = gm.weights_
    if V is None and ind_V is None:
        V, V_invT, ind_V, V_lam_sqr = _sqr(C_obs, cond_thresh=cond_thresh)

    if H is None:
        q_y = Q - P
    if method_condition == 'square_root':
        # square root implementation following Angus Andrews:
        # A Square Root Formulation of the Kalman Covariance Equations
        Wp_k = solve_triangular(gm.precisions_cholesky_[k, ...], np.eye(Q)).T
        if H is None:
            Z_k = np.transpose(Wp_k)[:, q_y:]
            U_k, U_invT_k, ind_U_k, U_k_lam_sqr = _sqr(
                Sigma[k, q_y:, q_y:] + C_obs, cond_thresh=cond_thresh)
            y_k_prior = mu[k, q_y:]
        UpV_k_inv = np.linalg.pinv(U_k + V, rcond=cond_thresh)
        UPD_k = np.eye(Q)[np.newaxis, ...] - np.einsum(
            'qb, mbc, mcd, ed -> mqe', Z_k, U_invT_k, UpV_k_inv, Z_k, optimize=True)
        # W_k neither triangular nor symmetric
        W_k = np.einsum('qb, mbc -> mqc', Wp_k, UPD_k, optimize=True)
        Sigma_p_k = np.einsum('mqb, mcb -> mqc', W_k, W_k, optimize=True)
        mu_p_k = mu[np.newaxis, k, :] + np.einsum(
            'pa, ab, mbc, mdc, md -> mp', Wp_k, Z_k, U_invT_k, U_invT_k, y_obs - y_k_prior,
            optimize=True)
        logdetfac, normfac = _nondata_terms_mvnormal(
            invert_nonzero(U_k_lam_sqr, ind_zero=ind_U_k['singular']),
            ind_singular=ind_U_k['singular'])
        prod = np.einsum('mqp, mq -> mp', U_invT_k, y_obs - y_k_prior)
        maha = -0.5 * np.sum(prod ** 2, axis=1)
        logpi_p_k = np.log(pi[k]) + maha + logdetfac + normfac
    elif method_condition == 'full':
        if H is None:
            y_k_prior = mu[k, q_y:]
            Sigma_k_off = Sigma[k, :, q_y:]
            Sigma_k_obs_inv = np.linalg.pinv(Sigma[k, q_y:, q_y:] + C_obs, rcond=cond_thresh)
            Sigma_k_obs = Sigma[k, q_y:, q_y:] + C_obs
        mu_p_k = mu[np.newaxis, k, :] + np.einsum(
            'pb, mbc, mc -> mp', Sigma_k_off, Sigma_k_obs_inv, y_obs - y_k_prior)
        Sigma_p_k = Sigma[np.newaxis, k, ...] - np.einsum(
            'pb, mbc, dc -> mpd', Sigma_k_off, Sigma_k_obs_inv, Sigma_k_off)
        log_p_y_obs = [multivariate_normal.logpdf(
                y_obs[m, ...], y_k_prior, Sigma_k_obs[m, ...], allow_singular=True) 
            for m in range(M)]
        logpi_p_k = np.log(pi[k]) + log_p_y_obs
    else:
        raise NotImplementedError

    mu_p_k[ind_V['invalid'], ...] = np.nan
    Sigma_p_k[ind_V['invalid'], ...] = np.nan
    logpi_p_k[ind_V['invalid'], ...] = np.nan
    return mu_p_k, Sigma_p_k, logpi_p_k

def posterior_gm_mvnormal(y_obs, C_obs, gm, H=None, cond_thresh=1e-6, 
                          method_condition='square_root'):
    # assumes last P dimensions in gm prior random variable are the surface positions

    K = len(gm.weights_)  # number of components
    P = y_obs.shape[1]
    M = y_obs.shape[0]
    Q = gm.means_.shape[1]  # dimension of prior RV considered
    assert C_obs.shape == (M, P, P)

    V, V_invT, ind_V, V_lam_sqr = _sqr(C_obs, cond_thresh=cond_thresh)

    logpi_p = np.empty((M, K))
    mu_p = np.empty((M, K, Q))
    Sigma_p = np.empty((M, K, Q, Q))
    
    for k in range(K):
        mu_p[:, k, ...], Sigma_p[:, k, ...], logpi_p[:, k] = _condition_gm(
            y_obs, C_obs, gm, k=k, V=V, ind_V=ind_V, cond_thresh=cond_thresh, 
            method_condition=method_condition)
    
    ind_invalid = np.any(np.isnan(logpi_p), axis=1)
    
    logpi_p[ind_invalid, ...] = 0.0    
    pi_p = np.exp(logpi_p - sumlogs(logpi_p, axis=1)[..., np.newaxis])
    mu_p[ind_invalid, ...] = np.nan
    Sigma_p[ind_invalid, ...] = np.nan
    pi_p[ind_invalid, ...] = np.nan
    
    return mu_p, Sigma_p, pi_p


if __name__ == '__main__':
    Q = 20
    P = 9
    N = 1000
    M = 200
    Sigma0 = np.eye(Q)

    C_obs = np.stack([np.diag(np.arange(P) + 1) + 0.0 * np.ones((P, P))] * M, axis=0)
    C_obs[1, 0, 0] = 1e-9
    C_obs[2, 0, 0] = -1

    sigma2s = (1.0, 3.0)
    pis = (0.3, 0.7)
    mus = (np.zeros(Q), 2 * np.ones(Q))

    rs = np.random.RandomState(seed=1)
    samples = []
    for jcomp in np.arange(len(sigma2s)):
        samples.append(
            rs.multivariate_normal(mean=mus[jcomp], cov=sigma2s[jcomp] * Sigma0,
                                   size=(int(pis[jcomp] * N),)))
    samples = np.concatenate(samples, axis=0)

    gm = fit_gaussian_mixture(samples, random_state=rs, n_init=2)
    y_obs = np.ones((M, P))

    import timeit

    testrun = lambda: posterior_gm_mvnormal(y_obs, C_obs, gm)
    print(timeit.timeit(stmt=testrun, number=1))
#     import matplotlib.pyplot as plt
#     plt.scatter(samples[:, 0], samples[:, 1])
#     plt.show()
