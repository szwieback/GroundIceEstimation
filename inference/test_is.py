'''
Created on Feb 27, 2020

@author: simon
'''
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from inference.psis import psislw, sumlogs
from inference.psis_vector import psislw as psislw2
# is faster

norm_proposal = (0.0, 0.4)
norm_true = (0.0, 1.0)
M = 100**2
N = 1000

def logweight(sample):
    lf = norm.logpdf(sample, loc=norm_true[0], scale=(norm_true[1])**(1/2))
    lg = norm.logpdf(sample, loc=norm_proposal[0], scale=(norm_proposal[1])**(1/2))
    return lf - lg

def normalized_weight(lw, keep_logarithmic=False):
    lw_ = lw - sumlogs(lw, axis=0)
    if not keep_logarithmic:
        lw_ = np.exp(lw_)
    return lw_

rs = np.random.RandomState(seed=1)

sample = rs.normal(loc=norm_proposal[0], scale=(norm_proposal[1])**(1/2), 
                            size=(N,M))
lw = logweight(sample)
# plt.hist(np.exp(lw), cumulative=False)
# plt.show()
import timeit
start_time = timeit.default_timer()
lw_out, kss = psislw(lw)
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
lw_out2, kss2 = psislw2(lw)
print(timeit.default_timer() - start_time)

print(np.nanpercentile(np.abs(lw_out-lw_out2), [5, 25, 50, 75, 95]))

# assert np.max(np.abs(kss-kss2)) < 1e-6
# assert np.max(np.abs(lw_out-lw_out2)) < 1e-6
# 
# power = 1
# m_raw = np.sum(sample**power/N, axis=0)
# m_is = np.sum(sample**power*normalized_weight(lw), axis=0)
# m_psis = np.sum(sample**power*normalized_weight(lw_out), axis=0)
# 
# print(np.std(m_is), np.std(m_psis))
# print(np.median(kss))

# plt.scatter(normalized_weight(lw), normalized_weight(lw_out))
# plt.show()
