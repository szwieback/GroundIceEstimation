'''
Created on Apr 30, 2020

@author: simon
'''
from cython.view cimport array as cvarray
import numpy as np

def stefan(dailytemp):
    k0 = 0.4
    ik = lambda yg: np.ones_like(yg)
    depth = 3.0
    dy = 0.001
    L_v = 3.34e8
    yg = np.arange(0, depth, step=dy)
    cdef Py_ssize_t Ng = yg.shape[0]
    cdef Py_ssize_t Nt = np.array(dailytemp).shape[0]
    wg = 0.4 * np.ones_like(yg)
    eg = 0.1 * np.ones_like(yg)
    ikg = ik(yg)
    Lg = L_v * (wg + eg)
    sg = np.cumsum(eg * dy)
    upsg = yg - sg
    k0d = k0 * 3600 * 24
    fac = 1e-9

    tterm = np.cumsum((k0d * fac) * np.array(dailytemp))
    yterm = np.cumsum(ikg * (Lg * fac) * upsg * dy)

    yf = np.ones_like(tterm) * -1
    s = np.zeros_like(tterm)

    cdef Py_ssize_t nt = 0
    cdef Py_ssize_t ny = 0
    cdef double [:] yf_v = yf
    cdef double [:] s_v = s
    cdef double [:] tterm_v = tterm
    cdef double [:] yterm_v = yterm

    while nt < Nt:
        if tterm_v[nt] >= yterm_v[ny]:
            ny += 1
            if ny == Ng:
                raise ValueError('depth not big enough')
        else:
            yf[nt] = ny * dy
            s[nt] = sg[ny]
            nt += 1
    return s, yf

def stefan_ens(dailytemp_ens):
    k0 = 0.4
    ik = lambda yg: np.ones_like(yg)
    depth = 3.0
    dy = 0.001
    L_v = 3.34e8
    yg = np.arange(0, depth, step=dy)
    cdef Py_ssize_t Ng = yg.shape[0]
    cdef Py_ssize_t Nt = dailytemp_ens.shape[1]
    cdef Py_ssize_t Ne = dailytemp_ens.shape[0]
    wg = 0.4 * np.ones_like(yg)
    eg = 0.1 * np.ones_like(yg)
    ikg = ik(yg)
    Lg = L_v * (wg + eg)
    sg = np.cumsum(eg * dy)
    upsg = yg - sg
    sg = sg
    k0d = k0 * 3600 * 24
    fac = 1e-9

    tterm = np.cumsum((k0d * fac) * dailytemp_ens, axis=1)
    yterm = np.zeros((Ne, Ng))
    yterm[:, :] = np.cumsum(ikg * (Lg * fac) * upsg * dy)[np.newaxis, :]

    yf = np.ones_like(tterm) * -1
    s = np.zeros_like(tterm)
    
    cdef Py_ssize_t nt = 0
    cdef Py_ssize_t ny = 0
    cdef double [:, :] yf_v = yf
    cdef double [:, :] s_v = s
    cdef double [:,:] tterm_v = tterm
    cdef double [:,:] yterm_v = yterm

    for ne in range(Ne):
        nt = 0
        ny = 0
        while nt < Nt:
            if tterm_v[ne, nt] >= yterm_v[ne, ny]:
                ny += 1
                if ny == Ng:
                    raise ValueError('depth not big enough')
            else:
                yf[ne, nt] = ny * dy
                s[ne, nt] = sg[ny]
                nt += 1
    return s, yf

