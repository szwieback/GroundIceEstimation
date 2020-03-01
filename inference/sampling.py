'''
Created on Feb 28, 2020

@author: simon
'''
import numpy as np
from inference import psislw, sumlogs

def _sqr_eigen(C , invert=False, cond_thresh=1e-6):
    lam, U = np.linalg.eigh(C)
    maxlamabs = np.max(np.abs(lam), axis=1)
    minlam = np.min(lam, axis=1)
    ind_invalid = minlam < -maxlamabs * cond_thresh
    lam[ind_invalid, :] = 1
    ind_singular = np.logical_and(lam < maxlamabs[:, np.newaxis] * cond_thresh,
                                  lam >= -maxlamabs[:, np.newaxis] * cond_thresh)
    lam[ind_singular] = 1
    if invert:
        lam_sqr = lam ** (-0.5)
    else:
        lam_sqr = lam ** (0.5)
    lam_sqr[ind_singular] = 0.0
    ind = {'invalid': ind_invalid, 'singular': ind_singular}
    return lam_sqr, U, ind

def invert_nonzero(b, ind_zero=None, thresh=1e-9):
    if ind_zero is None:
        ind_zero = np.abs(b) < thresh
    b2 = b.copy()
    b2[ind_zero] = 1
    np.power(b2, -1, out=b2)
    b2[ind_zero] = 0.0
    return b2

def _nondata_terms_mvnormal(lam_isqr, ind_singular=None):
    if ind_singular is None:
        ind_singular = np.full(lam_isqr.shape, False)
    lam_isqr[ind_singular] = 1.0 # for log determinant 
    logdetfac = np.sum(np.log(lam_isqr), axis=1)
    lam_isqr[ind_singular] = 0.0
    P_eff = np.sum(np.logical_not(ind_singular), axis=1)
    normfac = -(P_eff / 2) * np.log(2 * np.pi)
    return logdetfac, normfac
    
def lw_mvnormal(y_obs, C_obs, y_ref, cond_thresh=1e-6):
    # likelihood term corresponds to posterior/prior
    # y_obs: (M replicates, P observations over time, )
    # y_ref: (N samples, P_observations over time, )
    # returns non-normalized log weights
    # deals with (for practical purposes) singular C_obs

    P = y_obs.shape[1]
    M = y_obs.shape[0]
    N = y_ref.shape[0]
    assert C_obs.shape == (M, P, P)
    assert y_ref.shape[1] == P

    lw = np.zeros((N, M))
    lam_isqr, U, ind = _sqr_eigen(C_obs, cond_thresh=cond_thresh, invert=True)
    C_obs_isqrT = U * lam_isqr[:, np.newaxis, :]
    logdetfac, normfac = _nondata_terms_mvnormal(lam_isqr, ind_singular=ind['singular'])

    for n in np.arange(N):
        prod = np.einsum('mqp, mq -> mp', C_obs_isqrT, y_obs - y_ref[n, :][np.newaxis, :])
        maha = -0.5 * np.sum(prod ** 2, axis=1)
        lw[n, :] = maha + logdetfac + normfac

    lw[:, ind['invalid']] = np.nan
    return lw

if __name__ == '__main__':
    P = 11
    M = 200
    N = 1000
    C_obs = np.stack([np.diag(np.arange(P) + 1) + 0.0 * np.ones((P, P))] * M, axis=0)
    C_obs[1, 0, 0] = 1e-9
    C_obs[2, 0, 0] = -1

    y_obs = np.ones((M, P))
    y_ref = np.zeros((N, P)) + 2

    lw = lw_mvnormal(y_obs, C_obs, y_ref)
    from scipy.stats import multivariate_normal

    m = 0
    n = 2

    # check positive semidefinite: firx normfac (rank)
    print(lw[n, m])
    C = C_obs[m, ...]
    lpdf = multivariate_normal.logpdf(y_obs[n, :], y_ref[m, :], C_obs[m, ...],
                                      allow_singular=True)
    print(lpdf)

