'''
Created on Jun 16, 2023

@author: simon
'''
import numpy as np

def ind_TDD_exceedance(dailytemp, TDD):
    # dailytemp: from snow off [C]
    # TDD: iterable
    _ct = np.array(np.cumsum(dailytemp))
    inds = []
    for _TDD in TDD:
        inds.append(np.nonzero(_ct > _TDD)[0][0])
    return inds
        