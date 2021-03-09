'''
Created on Mar 8, 2021

@author: simon
'''
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

def stefan_ens_single(dailytemp_ens, params=params_default, k0ikupsQ=None):
    
    cdef Py_ssize_t Nt = dailytemp_ens.shape[1]
    cdef Py_ssize_t Ne = dailytemp_ens.shape[0]        
    cdef double dy = params['dy']
    yg = np.arange(0, params['depth'], step=dy)
    cdef Py_ssize_t Ng = yg.shape[0]
    
    Ct = params['Ct']
    e, w = _extract_stratigraphy(params, Ne, Ng)
    n_factor =  _extract_n_factor(params, Ne)
    k0, k0ik = _extract_conductivity(params, Ne, Ng)
    k0s = (k0 * (3600 * 24 * fac))  #scaled, per day
    
    Lg = (params['Lvw'] * (w + e)).astype(np.float32)
    sg = (np.cumsum(e * dy, axis=1)).astype(np.float32)
    ups = (yg[np.newaxis, :] - sg).astype(np.float32)
    k0ikups = (k0ik * ups).astype(np.float32)

    tterm = np.cumsum(n_factor[:, np.newaxis] * k0s[:, np.newaxis] * dailytemp_ens, axis=1)
    if k0ikupsQ is not None:
        tterm -= np.cumsum(k0ikupsQ, axis=1) * 3600 * 24 * fac
    tterm = tterm.astype(np.float32)
    yterm = np.cumsum(k0ikups * (Lg * fac) * dy, axis=1).astype(np.float32)

    yf = np.ones_like(tterm) * -1
    s = np.zeros_like(tterm)
    k0ikups_t = np.ones_like(tterm) #k0ik upsilon as function of time
    U_t = np.ones_like(tterm) # internal energy due to warming (above 0) in thawed part
    
    cdef Py_ssize_t nt = 0
    cdef Py_ssize_t ny = 0
    cdef float [:, :] yf_v = yf
    cdef float [:, :] s_v = s
    cdef float [:, :] tterm_v = tterm
    cdef float [:, :] yterm_v = yterm
    cdef float [:, :] k0ikups_t_v = k0ikups_t
    cdef float [:, :] U_t_v = U_t
    cdef float [:, :] dailytemp_ens_v = dailytemp_ens
    cdef float [:, :] ups_v = ups

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
                U_t_v[ne, nt] = 0.5 * Ct * dailytemp_ens_v[ne, nt] * ups_v[ne, ny]
                nt += 1
    return s, yf, k0ikups_t, U_t

def stefan_integral_balance_single(dailytemp_ens, params=params_default, steps=2):
    # simplified iterative approach based on Goodman's heat balance
    # for diffusion in frozen materials (uniform and constant properties)
    # and (assuming linear profile) storage changes in thawed part (also constant C)
    s, yf, k0ikups_t, U_t = stefan_ens_single(dailytemp_ens, params=params) # old values
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
        k0ikupsdUdt = np.zeros_like(k0ikupsQf)
        k0ikupsdUdt[:, 1:] = k0ikups_t[:, 1:] * np.diff(U_t, axis=1) / (24 * 3600) 
        k0ikupsdUdt[:, 0] = 0.0
#         print(np.mean(k0ikupsQf), np.mean(k0ikupsdUdt))
#         print(U_t[0, ::7] * 1e-6)
#         print(dailytemp_ens[0, ::7])
        k0ikupsQ = k0ikupsQf + k0ikupsdUdt
        # re-estimate freezing front progression keeping Q "losses" fixed
        s, yf, k0ikups_t, U_t = stefan_ens_single(
            dailytemp_ens, params=params, k0ikupsQ=k0ikupsQ)
        step = step + 1
    return s, yf

