'''
Created on Jan 19, 2023

@author: simon
'''
import numpy as np
import cython
from libc.math cimport exp, abs
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef quantile_bisection(float[:, :] vals, float[:, :] lw, float q, int steps):
    # lw must be normalized and 2D
    cdef Py_ssize_t M = lw.shape[0]
    cdef Py_ssize_t N = lw.shape[1]
    cdef Py_ssize_t R = vals.shape[1]
    
    qvals = np.zeros((M, R), dtype=np.float32)
    cdef float[:, ::1] qvals_view = qvals
    
    b_left = np.min(vals, axis=0)
    b_right = np.max(vals, axis=0)
    cdef float[::1] b_left_view = b_left
    cdef float[::1] b_right_view = b_right
    cdef float val_left, val_right

    vals_r = vals[:, 0].copy()
    cdef float[::1] vals_r_view = vals_r 
    qvals_m = np.zeros(R, dtype=np.float32)
    cdef float[::1] qvals_m_view = qvals_m     
    wn = lw[0, :].copy()
    cdef float[::1] wn_view = wn
    cdef float rsum

    cdef Py_ssize_t m, n, r, step
    for m in range(M):         
        for n in range(N):
            wn_view[n] = exp(lw[m, n])
        for r in range(R):
            for n in range(N):
                vals_r_view[n] = vals[n, r]
            val_left = b_left_view[r]
            val_right = b_right_view[r]
            step = 0
            while step < steps:
                val_mr = 0.5 * (val_left + val_right)
                rsum = 0.0
                for n in range(N):                    
                    if vals_r_view[n] < val_mr:
                        rsum += wn_view[n]
                if rsum < q:
                    val_left = val_mr
                else:
                    val_right = val_mr
                step += 1
            qvals_m_view[r] = 0.5 * (val_left + val_right)

        for r in range(R):
            qvals_view[m, r] = qvals_m_view[r]
    return qvals
