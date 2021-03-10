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
        return ("Simulated thaw depth exceeds computational domain. Increase depth.")

def extract_ensemble(dailytemp, params, balance=False):
    ens = {}
    Ng = np.arange(0, params['depth'], step=params['dy']).shape[0]    
    if len(dailytemp.shape) == 2:
        Ne = dailytemp.shape[0]
    elif 'e' in params:
        Ne = params['e'].shape[0]
    else:
        Ne = 1
    ind = np.arange(Ne)
    def _fill(field, val, size):
        if field not in params:
            filled = (val * np.ones(size))
        else:
            try:
                filled = params[field][ind, ...]
            except:
                filled = params[field] * np.ones(len(ind))
        return filled.astype(np.float32)
    if balance:
        ens['Cf'] = _fill('Cf', 1e6, (Ne,))
        ens['kf'] = _fill('kf', 1.9, (Ne,))
        ens['Tf'] = _fill('Tf', -4, (Ne,))
    else:
        if len(dailytemp.shape) == 2:
            ens['dailytemp'] = dailytemp[ind, :].astype(np.float32)
        else:
            ens['dailytemp'] = np.zeros((Ne, len(dailytemp)), dtype=np.float32)
            ens['dailytemp'][:, :] = np.array(dailytemp)[np.newaxis, :]
        ens['n_factor'] = _fill('n_factor', 0.9, (Ne,))
        ens['e'] = _fill('e', 0.1, (Ne, Ng))
        ens['w'] = _fill('w', 0.4, (Ne, Ng))
        ens['k0'] = _fill('k0', 0.4, (Ne,))
        ens['k0ik'] = _fill('k0ik', 1.0, (Ne, Ng))
        ens['Ct'] = _fill('Ct', 1e6, (Ne,))
    return ens

def stefan_initialize(dailytemp_ens, params, k0ikupsQ=None):
    yg = np.arange(0, params['depth'], step=params['dy'])  
    Ng = yg.shape[0]
    ens = extract_ensemble(dailytemp_ens, params)
    k0s = (ens['k0'] * (3600 * 24 * fac))  #scaled, per day
    
    Lg = (params['Lvw'] * (ens['w'] + ens['e'])).astype(np.float32)
    sg = (np.cumsum(ens['e'] * params['dy'], axis=1)).astype(np.float32)
    ups = (yg[np.newaxis, :] - sg).astype(np.float32)
    k0ikups = (ens['k0ik'] * ups).astype(np.float32)

    tterm = np.cumsum(
        ens['n_factor'][:, np.newaxis] * k0s[:, np.newaxis] * ens['dailytemp'], axis=1)
    if k0ikupsQ is not None:
        tterm -= np.cumsum(k0ikupsQ, axis=1) * (3600 * 24 * fac)
    tterm = tterm.astype(np.float32)
    yterm = np.cumsum(k0ikups * (Lg * fac) * params['dy'], axis=1).astype(np.float32)

    yf = np.ones_like(tterm) * -1
    s = np.zeros_like(tterm)
    k0ikups_t = np.ones_like(tterm) #k0ik upsilon as function of time
    U_t = np.ones_like(tterm) # internal energy due to warming (above 0) in thawed part
    ini = {'yf': yf, 's': s, 'tterm': tterm, 'yterm': yterm, 'k0ikups_t': k0ikups_t,
           'U_t': U_t, 'dailytemp': ens['dailytemp'], 'ups': ups, 'k0ikups': k0ikups,
           'sg': sg, 'Ct': ens['Ct']}
    return ini

def stefan_ens(dailytemp, params=params_default, k0ikupsQ=None):
    ini = stefan_initialize(
        dailytemp, params, k0ikupsQ=k0ikupsQ)
    s, yf = ini['s'], ini['yf']
    k0ikups_t, U_t = ini['k0ikups_t'], ini['U_t']
    dailytemp_ens = ini['dailytemp']
    cdef float [:, :] yf_v = yf
    cdef float [:, :] s_v = s
    cdef float [:, :] k0ikups_t_v = k0ikups_t
    cdef float [:, :] U_t_v = U_t

    cdef Py_ssize_t Nt = dailytemp_ens.shape[1]
    cdef Py_ssize_t Ne = dailytemp_ens.shape[0]        
    cdef double dy = params['dy']
    cdef Py_ssize_t Ng = int(params['depth'] / dy)    
    cdef Py_ssize_t nt = 0
    cdef Py_ssize_t ny = 0
    cdef float [:, :] tterm_v = ini['tterm']
    cdef float [:, :] yterm_v = ini['yterm']
    cdef float [:, :] dailytemp_ens_v = dailytemp_ens
    cdef float [:, :] ups_v = ini['ups']
    cdef float [:, :] k0ikups_v = ini['k0ikups']
    cdef float [:, :] sg_v = ini['sg']
    cdef float [:] Ct_v = ini['Ct']
    
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
                s_v[ne, nt] = sg_v[ne, ny]
                k0ikups_t_v[ne, nt] = k0ikups_v[ne, ny]
                U_t_v[ne, nt] = 0.5 * Ct_v[ne] * dailytemp_ens_v[ne, nt] * ups_v[ne, ny]
                nt += 1
    return s, yf, k0ikups_t, U_t

def stefan_integral_balance(
        dailytemp_ens, params=params_default, steps=2):
    # simplified iterative approach based on Goodman's heat balance
    # for diffusion in frozen materials (uniform and constant properties)
    # and (assuming linear profile) storage changes in thawed part (also constant C)

    # old values
    s, yf, k0ikups_t, U_t = stefan_ens(dailytemp_ens, params=params) 
    
    ens_balance = extract_ensemble(dailytemp_ens, params, balance=True)
    t = np.arange(1, 1 + yf.shape[1]) * 3600 * 24 # in seconds
    alphaf = (ens_balance['kf'] / ens_balance['Cf'])
    step = 0
    while step < steps:
        # estimate a and w  as function of t
        a = - 12 * alphaf[:, np.newaxis] * t[np.newaxis, :] * yf ** (-2)
        w = np.sqrt((9 / 4) - a) - (3 / 2) # w should be time invariant (neglect derivative)
        # estimate heat flux Qf into frozen material (function of time)
        Qf = -2 * (ens_balance['kf'] * ens_balance['Tf'])[:, np.newaxis] / (w * yf)
        # compute k0ikQf as function of time
        k0ikupsQf = k0ikups_t * Qf
        k0ikupsdUdt = np.zeros_like(k0ikupsQf)
        k0ikupsdUdt[:, 1:] = k0ikups_t[:, 1:] * np.diff(U_t, axis=1) / (24 * 3600) 
        k0ikupsdUdt[:, 0] = 0.0
        k0ikupsQ = k0ikupsQf + k0ikupsdUdt
        # re-estimate freezing front progression keeping Q "losses" fixed
        s, yf, k0ikups_t, U_t = stefan_ens(
            dailytemp_ens, params=params, k0ikupsQ=k0ikupsQ)
        step = step + 1
    return s, yf
    
