# Adapts code from:
"""Pareto smoothed importance sampling (PSIS)
This module implements Pareto smoothed importance sampling (PSIS) and PSIS
leave-one-out (LOO) cross-validation for Python (Numpy).
Included functions
------------------
psisloo
    Pareto smoothed importance sampling leave-one-out log predictive densities.
psislw
    Pareto smoothed importance sampling.
gpdfitnew
    Estimate the paramaters for the Generalized Pareto Distribution (GPD).
gpinv
    Inverse Generalised Pareto distribution function.
sumlogs
    Sum of vector where numbers are represented by their logarithms.
References
----------
Aki Vehtari, Andrew Gelman and Jonah Gabry (2017). Practical
Bayesian model evaluation using leave-one-out cross-validation
and WAIC. Statistics and Computing, 27(5):1413–1432.
doi:10.1007/s11222-016-9696-4. https://arxiv.org/abs/1507.04544
Aki Vehtari, Andrew Gelman and Jonah Gabry (2017). Pareto
smoothed importance sampling. https://arxiv.org/abs/arXiv:1507.02646v5
"""
"""
Copyright 2017 Aki Vehtari, Tuomas Sivula
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. """

import numpy as np

def psislw(lw, Reff=1.0):
    # operates on log weights lw (normalization not required)
    # lw: 2 dimensional; dim 1: replicates, dim 2: samples [i.e. different to original]
    assert lw.ndim == 2
    M, N = lw.shape

    lw_out = np.copy(lw)
    k_min = 1 / 3

    # precalculate constants
    cutoff_ind = N - int(np.ceil(min(0.2 * N, 3 * np.sqrt(N / Reff))))
    N_large = N - cutoff_ind
    lw_out -= np.max(lw_out, axis=1)[:, np.newaxis]
    ind_sort = np.argsort(lw_out, axis=1)
    if N_large <= 4:
        k = np.zeros(M) + np.nan
    else:
        # those > cutoff; a copy
        lw_out_large = lw_out[(np.arange(M)[:, np.newaxis], ind_sort[:, cutoff_ind:])]
        cutoff = lw_out[(np.arange(M), ind_sort[:, cutoff_ind - 1])]
        expcutoff = np.exp(cutoff)
        w_out_large = np.exp(lw_out_large) - expcutoff[:, np.newaxis]
        k, sigma = gpdfit(w_out_large)
        # find out where smoothing does not makes sense
        ind_nonsmooth_m = np.nonzero(
            np.logical_or(k < k_min, np.logical_not(np.isfinite(k))))
        sti = np.arange(0.5, N_large) / N_large
        qq = np.log(gpinv(sti, k, sigma) + expcutoff[:, np.newaxis])
        qq[ind_nonsmooth_m[0], :] = lw_out_large[ind_nonsmooth_m[0], :]
        lw_out[(np.arange(M)[:, np.newaxis], ind_sort[:, cutoff_ind:])] = qq
        lw_out[lw_out > 0] = 0
    lw_out -= sumlogs(lw_out, axis=1)[:, np.newaxis]
    return lw_out, k

def gpdfit(w):
    # w assumed to be sorted, 2 dimensional
    N = w.shape[1]
    prior = 3
    G = 30 + int(np.sqrt(N))
    b = (1 - np.sqrt(G / (np.arange(1, G + 1, dtype=float) - 0.5)))[np.newaxis, :]
    bs = (b / (prior * w[:, int(N / 4 + 0.5) - 1][:, np.newaxis]))
    bs += 1 / w[:, -1][:, np.newaxis]
    ks = np.mean(np.log1p(-bs[:, np.newaxis, :] * w[..., np.newaxis]), axis=1)
    L = N * (np.log(-bs / ks) - ks - 1)
    wsum = np.empty_like(L)
    for jG in range(G):
        wsum[:, jG] = 1 / (np.sum(np.exp(L - L[:, jG][:, np.newaxis]), axis=1))
    wsum /= np.sum(wsum, axis=1)[..., np.newaxis]
    # posterior mean for b
    b = np.sum(bs * wsum, axis=1)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterisation.
    k = np.mean(np.log1p(-b[:, np.newaxis] * w), axis=1)
    # estimate for sigma
    sigma = -k / b * N / (N - 0)
    # weakly informative prior for k
    a = 10
    k = k * N / (N + a) + a * 0.5 / (N + a)

    return k, sigma

def gpinv(p, k, sigma):
    """Inverse Generalised Pareto distribution function."""
    assert p.ndim == 1
    assert k.ndim == 1
    assert k.shape == sigma.shape
    x = (sigma[:, np.newaxis] * np.expm1(-k[:, np.newaxis] * np.log1p(-p[np.newaxis, :]))
         / k[:, np.newaxis])
    x2 = -sigma[:, np.newaxis] * np.log1p(-p[np.newaxis, :])
    x = np.where((np.abs(k) < 10 * np.finfo(float).eps)[:, np.newaxis], x2, x)
    return x

def sumlogs(x, axis=None, out=None):
    """Sum of vector where numbers are represented by their logarithms.
    Calculates ``np.log(np.sum(np.exp(x), axis=axis))`` in such a fashion that
    it works even when elements have large magnitude.
    """
    maxx = x.max(axis=axis, keepdims=True)
    xnorm = x - maxx
    np.exp(xnorm, out=xnorm)
    out = np.sum(xnorm, axis=axis, out=out)
    if isinstance(out, np.ndarray):
        np.log(out, out=out)
    else:
        out = np.log(out)
    out += np.squeeze(maxx)
    return out

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
    lam_isqr[ind_singular] = 1.0  # for log determinant
    logdetfac = np.sum(np.log(lam_isqr), axis=1)
    lam_isqr[ind_singular] = 0.0
    P_eff = np.sum(np.logical_not(ind_singular), axis=1)
    normfac = -(P_eff / 2) * np.log(2 * np.pi)
    return logdetfac, normfac

def _normalize(lw, normalize=False):
    if normalize:
        lw_ = lw - sumlogs(lw, axis=1)[:, np.newaxis]
    else:
        lw_ = lw
    return lw_


def lw_mvnormal(y_obs, C_obs, y_ref, cond_thresh=1e-6, normalize=False):
    # likelihood term corresponds to posterior/prior
    # y_obs: (M replicates, P observations over time, )
    # y_ref: (N samples, P_observations over time, )
    # returns log weights; default: not normalized
    # deals with (for practical purposes) singular C_obs

    P = y_obs.shape[1]
    M = y_obs.shape[0]
    N = y_ref.shape[0]
    assert C_obs.shape == (M, P, P)
    assert y_ref.shape[1] == P

    lw = np.zeros((M, N))
    lam_isqr, U, ind = _sqr_eigen(C_obs, cond_thresh=cond_thresh, invert=True)
    C_obs_isqrT = U * lam_isqr[:, np.newaxis, :]
    logdetfac, normfac = _nondata_terms_mvnormal(lam_isqr, ind_singular=ind['singular'])

    for n in np.arange(N):
        prod = np.einsum('mqp, mq -> mp', C_obs_isqrT, y_obs - y_ref[n, :][np.newaxis, :])
        maha = -0.5 * np.sum(prod ** 2, axis=1)
        lw[:, n] = maha + logdetfac + normfac

    lw = _normalize(lw, normalize=normalize)
    lw[ind['invalid'], :] = np.nan

    return lw

def expectation(vals, lw, normalize=False):
    # vals: samples, val dimension (e.g. depth)
    # lw: replicates, samples
    lw_ = _normalize(lw, normalize=normalize)
    x = np.einsum('ij, jk -> ik', np.exp(lw_), vals)
    return x

def quantile(vals, lw, q, steps=100, normalize=False):
    lw_ = _normalize(lw, normalize=normalize)
    valmin, valmax = np.min(vals), np.max(vals)
    valq = np.zeros((vals.shape[1],)) + np.nan
    vald = np.ones((vals.shape[1],))
    for valtrial in np.linspace(valmin, valmax, steps):
        vals_ind = (vals < valtrial).astype(np.float64)
        dtrial = np.abs(expectation(vals_ind, lw_) - q)[0, ...]
        np.putmask(valq, dtrial < vald, valtrial)
        np.putmask(vald, dtrial < vald, dtrial)
    return valq


if __name__ == '__main__':
    pass

