'''
Created on Feb 27, 2020

@author: simon
'''
import numpy as np
from scipy.stats import norm
import unittest

from inference.psis import psislw as psislw2
from inference.isi import psislw, sumlogs

norm_proposal = (0.0, 0.4)
norm_true = (0.0, 1.0)
M = 100
N = 1000

def logweight(sample):
    lf = norm.logpdf(sample, loc=norm_true[0], scale=(norm_true[1]) ** (1 / 2))
    lg = norm.logpdf(sample, loc=norm_proposal[0], scale=(norm_proposal[1]) ** (1 / 2))
    return lf - lg

def normalized_weight(lw, keep_logarithmic=False):
    lw_ = lw - sumlogs(lw, axis=1)[:, np.newaxis]
    if not keep_logarithmic:
        lw_ = np.exp(lw_)
    return lw_

class PSISTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rs = np.random.RandomState(seed=1)
        cls.sample = rs.normal(loc=norm_proposal[0], scale=(norm_proposal[1]) ** (1 / 2),
                               size=(M, N))

    def setUp(self):
        self.lw = normalized_weight(logweight(self.sample), keep_logarithmic=True)
        self.lw_out, self.kss = psislw(self.lw)
        lw_out2T, self.kss2 = psislw2(self.lw.T)
        self.lw_out2 = lw_out2T.T

    def test_logweights(self):
        # by comparison with original python code
        self.assertLess(np.max(np.abs(self.lw_out - self.lw_out2)), 1e-9)
    
    def test_kss(self):
        self.assertLess(np.max(np.abs(self.kss - self.kss2)), 1e-9)
        
    def test_shape(self):
        self.assertEqual(self.sample.shape, self.lw_out.shape)
        self.assertEqual(self.kss.shape[0], self.sample.shape[0])
        self.assertEqual(self.kss.ndim, 1)

    def test_normalized(self):
        self.assertLess(np.max(np.abs(sumlogs(self.lw_out, axis=1))), 1e-9)

if __name__ == '__main__':
    unittest.main()



