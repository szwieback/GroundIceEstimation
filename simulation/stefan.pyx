'''
Created on Apr 30, 2020

@author: simon
'''
from cython.view cimport array as cvarray
import numpy as np

from stratigraphy import params_default

fac = 1e-9 # scale integral to avoid huge numbers [printing a pain]


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

def stefan_ens(dailytemp_ens, params=params_default, k0ikupsQf=None):
    
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
    k0ikups = k0ik * (yg[np.newaxis, :] - sg)

    tterm = np.cumsum(n_factor[:, np.newaxis] * k0s[:, np.newaxis] * dailytemp_ens, axis=1)
    if k0ikupsQf is not None:
        tterm -= np.cumsum(k0ikupsQf, axis=1) * 3600 * 24 * fac
    yterm = np.cumsum(k0ikups * (Lg * fac) * dy, axis=1)

    yf = np.ones_like(tterm) * -1
    s = np.zeros_like(tterm)
    k0ikups_t = np.ones_like(tterm) #k0ik as function of time
    
    cdef Py_ssize_t nt = 0
    cdef Py_ssize_t ny = 0
    cdef double [:, :] yf_v = yf
    cdef double [:, :] s_v = s
    cdef double [:, :] tterm_v = tterm
    cdef double [:, :] yterm_v = yterm
    cdef double [:, :] k0ikups_t_v = k0ikups_t

    for ne in range(Ne):
        nt = 0
        ny = 1
        while nt < Nt:
            if tterm_v[ne, nt] >= yterm_v[ne, ny]:
                ny += 1
                if ny == Ng:
                    raise DepthExceedanceException()
            else:
                yf_v[ne, nt] = ny * dy
                s_v[ne, nt] = sg[ne, ny]
                k0ikups_t_v[ne, nt] = k0ikups[ne, ny]
                nt += 1
    return s, yf, k0ikups_t

def stefan_integral_balance(dailytemp_ens, params=params_default, steps=2):
    # simplified iterative approach based on Goodman's heat balance 
    # for diffusion in frozen materials (uniform and constant properties)
    s, yf, k0ikups_t = stefan_ens(dailytemp_ens, params=params) # old values
    t = np.arange(1, 1 + yf.shape[1]) * 3600 * 24 # in seconds
    alphaf = (params['kf'] / params['Cf'])
    step = 0
    while step < steps:
        # estimate a and w  as function of t
        a = - 12 * alphaf[:, np.newaxis] * t[np.newaxis, :] * yf ** (-2)
        w = np.sqrt((9 / 4) - a) - (3 / 2) # w should be time invariant (neglect derivative)
        # estimate heat flux Qf into frozen material (function of time)
        Qf = -2 * (params['kf'] * params['Tf'])[:, np.newaxis] / (w * yf)
        # compute k0ikQf as function of time
        k0ikupsQf = k0ikups_t * Qf
        #reestimate freezing front progression keeping Qf fixed
        s, yf, k0ikups_t = stefan_ens(dailytemp_ens, params=params, k0ikupsQf=k0ikupsQf)
        step = step + 1
    return s, yf

