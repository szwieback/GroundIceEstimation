'''
Created on Apr 30, 2020

@author: simon
'''
from cython.view cimport array as cvarray
import numpy as np

constants = {'Lvw': 3.34e8,  'km': 3.80, 'ko': 0.25, 'ki': 2.20, 'kw': 0.57, 'ka': 0.024,}

class DepthExceedanceException(Exception):
    
    def __init__(self):
        pass
    
    def __str__(self):
        return ('Simulated thaw depth exceeds computational domain. Increase depth.')

def _extract_stratigraphy(params, Ne, Ng):
    if 'e' not in params:
        e = 0.1 * np.ones((Ne, Ng))
    else:
        e = params['e']
    
    if 'w' not in params:
        w = 0.4 * np.ones((Ne, Ng))
    else:
        w = params['w']
    return e, w

def _extract_n_factor(params, Ne):
    if 'n_factor' not in params:
        n_factor = 0.9 * np.ones((Ne,))
    else:
        n_factor = params['n_factor']
    return n_factor

def _extract_conductivity(params, Ne, Ng):
    if 'k0' not in params:
        k0 = 0.4* np.ones((Ne,))
    else:
        k0 = params['k0']
    if 'k0ik' not in params:
        k0ik = np.ones((Ne, Ng))
    else:
        k0ik = params['k0ik']
    return k0, k0ik


fac = 1e-9 # scale integral to avoid huge numbers [printing a pain]
def stefan_ens(dailytemp_ens, params=constants):
    
    cdef Py_ssize_t Nt = dailytemp_ens.shape[1]
    cdef Py_ssize_t Ne = dailytemp_ens.shape[0]        
    cdef double dy = params['dy']
    yg = np.arange(0, params['depth'], step=dy)
    cdef Py_ssize_t Ng = yg.shape[0]

    e, w = _extract_stratigraphy(params, Ne, Ng)
    n_factor =  _extract_n_factor(params, Ne)
    k0, k0ik = _extract_conductivity(params, Ne, Ng)
    k0s = (k0 * 3600 * 24 * fac)  #scaled, per day
    
    Lg = params['Lvw'] * (w + e)
    sg = np.cumsum(e * dy, axis=1)
    upsg = yg[np.newaxis, :] - sg

    tterm = np.cumsum(n_factor[:, np.newaxis] * k0s[:, np.newaxis] * dailytemp_ens, axis=1)
    yterm = np.cumsum(k0ik * (Lg * fac) * upsg * dy, axis=1)

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

    