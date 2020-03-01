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
    # lw: 2 dimensional; dim 1: samples, dim 2: replicates
    assert lw.ndim == 2
    N, M = lw.shape

    lw_out = np.copy(lw)
    k_min = 1 / 3

    # precalculate constants
    cutoff_ind = N - int(np.ceil(min(0.2 * N, 3 * np.sqrt(N / Reff))))
    N_large = N - cutoff_ind
    lw_out -= np.max(lw_out, axis=0)[np.newaxis, :]
    ind_sort = np.argsort(lw_out, axis=0)
    if N_large <= 4:
        k = np.zeros(M) + np.nan
    else:
        # those > cutoff; a copy
        lw_out_large = lw_out[(ind_sort[cutoff_ind:, :], np.arange(M))]
        cutoff = lw_out[(ind_sort[cutoff_ind - 1, :], np.arange(M))]
        expcutoff = np.exp(cutoff)
        w_out_large = np.exp(lw_out_large) - expcutoff[np.newaxis, :]
#         k, sigma = gpdfitnew(w_out_large[:, 0], sort=False)
        k, sigma = gpdfitvec(w_out_large)
        # find out where smoothing does not makes sense
        ind_nonsmooth_m = np.nonzero(np.logical_or(k < k_min, np.logical_not(np.isfinite(k))))
#         ind_m = np.nonzero(np.logical_and(k >= k_min, np.isfinite(k)))
        sti = np.arange(0.5, N_large) / N_large
        qq = np.log(gpinv_vec(sti, k, sigma) + expcutoff)
        qq[:, ind_nonsmooth_m[0]] = lw_out_large[:, ind_nonsmooth_m[0]]
        lw_out[(ind_sort[cutoff_ind:, :], np.arange(M))] = qq
        lw_out[lw_out > 0] = 0
    lw_out -= sumlogs(lw_out, axis=0)[np.newaxis, :]
    return lw_out, k

def gpdfitvec(w):
    # w assumed to be sorted, 2 dimensional
    # axis 0 is samples
    N = w.shape[0]

    prior = 3
    m = 30 + int(np.sqrt(N))

    b = (1 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5)))[:, np.newaxis]
    bs = (b / (prior * w[int(N / 4 + 0.5) - 1, :]))
    bs += 1 / w[-1, :]
    ks = np.mean(np.log1p(-bs[:, np.newaxis, :] * w[np.newaxis, ...]), axis=1)

    L = N * (np.log(-bs / ks) - ks - 1)
    wsum = np.empty_like(L)
    for jm in range(m):
        wsum[jm, :] = 1/(np.sum(np.exp(L - L[jm, ...][np.newaxis, ...]), axis=0))

    wsum /= np.sum(wsum, axis=0)[np.newaxis, ...]
    # posterior mean for b
    b = np.sum(bs * wsum, axis=0)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterisation.
    k = np.mean(np.log1p((-b) * w), axis=0)

    # estimate for sigma
    sigma = -k / b * N / (N - 0)
    # weakly informative prior for k
    a = 10
    k = k * N / (N + a) + a * 0.5 / (N + a)

    return k, sigma

def gpinv_vec(p, k, sigma):
    """Inverse Generalised Pareto distribution function."""
    assert p.ndim == 1
    assert k.ndim == 1
    assert k.shape == sigma.shape
    x = (sigma[np.newaxis, ...] * np.expm1(-k[np.newaxis, ...] * np.log1p(-p[..., np.newaxis]))
         / k[np.newaxis, ...])
    x2 = -sigma[np.newaxis, ...] * np.log1p(-p[..., np.newaxis])
    x = np.where((np.abs(k) < 10 * np.finfo(float).eps)[np.newaxis, ...], x2, x)
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
