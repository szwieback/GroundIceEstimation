'''
Created on Apr 30, 2020

@author: simon
'''
from cython.view cimport array as cvarray
import numpy as np


params_ens_default = {'depth': 2.5, 'dy': 2.5e-3}
params_default = {'Lvwater': 3.34e8}

class DepthExceedanceException(Exception):
    
    def __init__(self):
        pass
    
    def __str__(self):
        return ('Simulated thaw depth exceeds computational domain. Increase depth.')

def _extract_stratigraphy(params_ens, Ne, Ng):
    if 'e' not in params_ens:
        e = 0.1 * np.ones((Ne, Ng))
    else:
        e = params_ens['e']
    
    if 'w' not in params_ens:
        w = 0.4 * np.ones((Ne, Ng))
    else:
        w = params_ens['w']
    return e, w

def _extract_n_factor(params_ens, Ne):
    if 'n_factor' not in params_ens:
         n_factor = 0.9 * np.ones((Ne,))
    else:
        n_factor = params_ens['n_factor']
    return n_factor

fac = 1e-9 # scale integral to avoid huge numbers [printing a pain]
def stefan_ens(
    dailytemp_ens, params_ens=params_ens_default, params=params_default):
    
    cdef Py_ssize_t Nt = dailytemp_ens.shape[1]
    cdef Py_ssize_t Ne = dailytemp_ens.shape[0]
    
    k0 = 0.4
    ik = lambda yg: np.ones_like(yg)
    
    cdef double dy = params_ens['dy']
    yg = np.arange(0, params_ens['depth'], step=dy)
    k0s = (k0 * 3600 * 24 * fac) * np.ones((Ne,)) #scaled, per day
    
    cdef Py_ssize_t Ng = yg.shape[0]

    e, w = _extract_stratigraphy(params_ens, Ne, Ng)
    n_factor =  _extract_n_factor(params_ens, Ne)
        
    ikg = ik(yg)
    Lg = params['Lvwater'] * (w + e)
    sg = np.cumsum(e * dy, axis=1)
    upsg = yg[np.newaxis, :] - sg

    tterm = np.cumsum(n_factor[:, np.newaxis] * k0s[:, np.newaxis] * dailytemp_ens, axis=1)
    yterm = np.cumsum(ikg * (Lg * fac) * upsg * dy, axis=1)

    yf = np.ones_like(tterm) * -1
    s = np.zeros_like(tterm)
    
    cdef Py_ssize_t nt = 0
    cdef Py_ssize_t ny = 0
    cdef double [:, :] yf_v = yf
    cdef double [:, :] s_v = s
    cdef double [:, :] tterm_v = tterm
    cdef double [:, :] yterm_v = yterm

    for ne in range(Ne):
        nt = 0
        ny = 0
        while nt < Nt:
            if tterm_v[ne, nt] >= yterm_v[ne, ny]:
                ny += 1
                if ny == Ng:
                    raise DepthExceedanceException()
            else:
                yf[ne, nt] = ny * dy
                s[ne, nt] = sg[ne, ny]
                nt += 1
    return s, yf

