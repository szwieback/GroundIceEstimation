'''
Created on Feb 29, 2020

@author: simon
'''
import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_normal

from inference.isi import _sqr_eigen, _nondata_terms_mvnormal, invert_nonzero, sumlogs

def fit_gaussian_mixture(
        samples, random_state=None, n_components=2, covariance_type='full', n_init=8, init_params='random',
        ):
    gm = GaussianMixture(
        random_state=random_state, n_components=n_components, covariance_type=covariance_type, n_init=n_init,
        init_params=init_params, verbose=0).fit(samples)
    assert gm.converged_

    return gm

def _sqr(A, cond_thresh=1e-6,):
    lam_sqr, Q, ind = _sqr_eigen(A, cond_thresh=cond_thresh)
    sqr = Q * lam_sqr[:, np.newaxis,:]
    lam_sqr[ind['singular']] = 1
    lam_isqr = lam_sqr ** (-1)
    lam_isqr[ind['singular']] = 0
    isqrT = Q * lam_isqr[:, np.newaxis,:]
    return sqr, isqrT, ind, lam_sqr

def _condition_gm(
        y_obs, C_obs, gm, k=0, V=None, ind_V=None, cond_thresh=1e-6,
        method_condition='square_root'):
    # method_condition: 'square_root' or 'full'
    # (latter does not handle nonpos-def matrices, mainly as external check)
    mu = gm.means_
    Sigma = gm.covariances_
    pi = gm.weights_

    P = y_obs.shape[1]
    M = y_obs.shape[0]
    Q = mu.shape[1]  # dimension of prior RV considered

    if V is None and ind_V is None:
        V, V_invT, ind_V, V_lam_sqr = _sqr(C_obs, cond_thresh=cond_thresh)

    q_y = Q - P
    if method_condition == 'square_root':
        # square root implementation following Angus Andrews:
        # A Square Root Formulation of the Kalman Covariance Equations
        Wp_k = solve_triangular(gm.precisions_cholesky_[k, ...], np.eye(Q)).T
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
        mu_p_k = mu[np.newaxis, k,:] + np.einsum(
            'pa, ab, mbc, mdc, md -> mp', Wp_k, Z_k, U_invT_k, U_invT_k, y_obs - y_k_prior,
            optimize=True)
        logdetfac, normfac = _nondata_terms_mvnormal(
            invert_nonzero(U_k_lam_sqr, ind_zero=ind_U_k['singular']),
            ind_singular=ind_U_k['singular'])
        prod = np.einsum('mqp, mq -> mp', U_invT_k, y_obs - y_k_prior)
        maha = -0.5 * np.sum(prod ** 2, axis=1)
        logpi_p_k = np.log(pi[k]) + maha + logdetfac + normfac
    elif method_condition == 'full':
        y_k_prior = mu[k, q_y:]
        Sigma_k_off = Sigma[k,:, q_y:]
        Sigma_k_obs = Sigma[k, q_y:, q_y:] + C_obs            
        Sigma_k_obs_inv = np.linalg.pinv(Sigma_k_obs, rcond=cond_thresh)
        mu_p_k = mu[np.newaxis, k,:] + np.einsum(
            'pb, mbc, mc -> mp', Sigma_k_off, Sigma_k_obs_inv, y_obs - y_k_prior)
        Sigma_p_k = Sigma[np.newaxis, k, ...] - np.einsum(
            'pb, mbc, dc -> mpd', Sigma_k_off, Sigma_k_obs_inv, Sigma_k_off)
        log_p_y_obs = [multivariate_normal.logpdf(
                y_obs[m, ...], y_k_prior, Sigma_k_obs[m, ...], allow_singular=True)
            for m in range(M)]
        logpi_p_k = np.log(pi[k]) + log_p_y_obs
    elif method_condition == 'unobserved':
        y_k_prior = mu[k, q_y:]
        Sigma_k_off = Sigma[k,:q_y, q_y:]
        Sigma_k_obs = Sigma[k, q_y:, q_y:] + C_obs            
        Sigma_k_obs_inv = np.linalg.pinv(Sigma_k_obs, rcond=cond_thresh)
        dy = y_obs - y_k_prior
        mu_p_k = mu[np.newaxis, k,:q_y] + np.einsum(
            'pb, mbc, mc -> mp', Sigma_k_off, Sigma_k_obs_inv, dy)
        Sigma_p_k = Sigma[np.newaxis, k, :q_y, :q_y] - np.einsum(
            'pb, mbc, dc -> mpd', Sigma_k_off, Sigma_k_obs_inv, Sigma_k_off)
        log_p_y_obs = - 0.5 * (
            P * np.log(2 * np.pi) + np.linalg.slogdet(Sigma_k_obs)[1]
            + np.einsum('mc, md, mcd -> m', dy, dy, Sigma_k_obs_inv))

        # log_p_y_obs2 = [multivariate_normal.logpdf(
        #         y_obs[m, ...], y_k_prior, Sigma_k_obs[m, ...], allow_singular=True)
        #     for m in range(M)]
        
        logpi_p_k = np.log(pi[k]) + log_p_y_obs

    else:
        raise NotImplementedError

    mu_p_k[ind_V['invalid'], ...] = np.nan
    Sigma_p_k[ind_V['invalid'], ...] = np.nan
    logpi_p_k[ind_V['invalid'], ...] = np.nan
    return mu_p_k, Sigma_p_k, logpi_p_k

def posterior_gm_mvnormal(
        y_obs, C_obs, gm, cond_thresh=1e-6, method_condition='square_root'):
    # assumes last P dimensions in gm prior random variable are the observations

    K = len(gm.weights_)  # number of components
    P = y_obs.shape[1]  # number of observations
    M = y_obs.shape[0]  # number of replicates
    Q = gm.means_.shape[1]  # dimension of prior RV considered
    Qout = Q if not 'unobserved' in method_condition else Q - P
    assert C_obs.shape == (M, P, P)

    V, V_invT, ind_V, V_lam_sqr = _sqr(C_obs, cond_thresh=cond_thresh)

    logpi_p = np.empty((K, M))
    mu_p = np.empty((K, M, Qout))
    Sigma_p = np.empty((K, M, Qout, Qout))

    for k in range(K):
        mu_p[k, ...], Sigma_p[k, ...], logpi_p[k, ...] = _condition_gm(
            y_obs, C_obs, gm, k=k, V=V, ind_V=ind_V, cond_thresh=cond_thresh,
            method_condition=method_condition)

    ind_invalid = np.any(np.isnan(logpi_p), axis=0)
    logpi_p[:, ind_invalid] = 0.0
    pi_p = np.exp(logpi_p - sumlogs(logpi_p, axis=0)[np.newaxis, ...])
    mu_p[:, ind_invalid] = np.nan
    Sigma_p[:, ind_invalid] = np.nan
    pi_p[:, ind_invalid] = np.nan
    return mu_p, Sigma_p, pi_p

def posterior_moments(mu_p, Sigma_p, pi_p):
    mu = np.einsum('ijk, ij -> jk', mu_p, pi_p)
    mu_dev = mu_p - mu[np.newaxis, ...]
    Sigma = np.einsum(
        'ijkl, ij -> jkl', Sigma_p + mu_dev[..., np.newaxis] * mu_dev[..., np.newaxis,:], pi_p)
    return mu, Sigma

if __name__ == '__main__':
    N = 2048  # samples
    P = 8
    H = np.linspace(1, 4, num=P)[np.newaxis,:]
    rs = np.random.RandomState(seed=5)
    parms = rs.uniform(low=0.0, high=0.8, size=(N, 1))
    y_ref = parms * H
    y_ref += rs.normal(scale=0.  , size=(N, P))

    # fit
    samples = np.concatenate((parms, y_ref), axis=-1)
    gm = fit_gaussian_mixture(samples, n_init=3, n_components=2)
    # observations
    M = 512
    parm0 = np.array([0.7])
    sigma_obs = 0.10
    y_obs = rs.normal(scale=sigma_obs, size=(M, P))
    y_obs[...] += parm0[np.newaxis,:] * H
    C_obs = np.zeros((M, P, P))
    C_obs[:, ...] = sigma_obs ** 2 * np.eye(P)[np.newaxis, ...]
    mu_p, Sigma_p, pi_p = posterior_gm_mvnormal(y_obs, C_obs, gm)
    mu_pu, Sigma_pu, pi_pu = posterior_gm_mvnormal(y_obs, C_obs, gm, method_condition='unobserved')  # need to cut this down


    # print(Sigma_pu[:, :, 0, 0] - Sigma_p[:, :, 0, 0])
    print(pi_p - pi_pu)
    # implement final estimate
    # mu, Sigma = posterior_moments(mu_p, Sigma_p, pi_p)
    # print(mu_p.shape, mu.shape)

    # replace log_p_y_obs
    # add square root smaller subset
    # refactor posterior function
    # check whether adding C_obs is permissible [should be]
    
