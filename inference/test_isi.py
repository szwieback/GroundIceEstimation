'''
Created on Feb 27, 2020

@author: simon
'''
import numpy as np
from scipy.stats import norm, multivariate_normal
import unittest

from inference.psis import psislw as psislw2
from inference.isi import psislw, sumlogs, lw_mvnormal

class PSISTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rs = np.random.RandomState(seed=1)
        cls.norm_proposal = (0.0, 0.4)
        cls.norm_true = (0.0, 1.0)
        cls.M = 100
        cls.N = 1000
        cls.sample = rs.normal(
            loc=cls.norm_proposal[0], scale=(cls.norm_proposal[1]) ** (1 / 2), 
            size=(cls.M, cls.N))

    def setUp(self):
        self.lw = self.normalized_weight(self.logweight(self.sample), keep_logarithmic=True)
        self.lw_out, self.kss = psislw(self.lw)
        lw_out2T, self.kss2 = psislw2(self.lw.T)
        self.lw_out2 = lw_out2T.T

    def logweight(self, sample):
        lf = norm.logpdf(
            sample, loc=self.norm_true[0], scale=(self.norm_true[1]) ** (1 / 2))
        lg = norm.logpdf(
            sample, loc=self.norm_proposal[0], scale=(self.norm_proposal[1]) ** (1 / 2))
        return lf - lg

    def normalized_weight(self, lw, keep_logarithmic=False):
        lw_ = lw - sumlogs(lw, axis=1)[:, np.newaxis]
        if not keep_logarithmic:
            lw_ = np.exp(lw_)
        return lw_

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

class MVNormalTestCase1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.P = 11
        cls.M = 3
        cls.N = 1000
        cls.cond_thresh = 1e-6
        cls.__set_C_obs()
        cls.y_obs = np.ones((cls.M, cls.P))
        cls.y_ref = np.zeros((cls.N, cls.P)) + 2
        
    @classmethod
    def __set_C_obs(cls):
        cls.C_obs = np.stack([np.diag(np.arange(cls.P) + 1)] * cls.M, axis=0)
        cls.C_obs[1, 0, 0] = cls.cond_thresh * 1e-3
        cls.C_obs[2, 0, 0] = -1

    def setUp(self):
        self.lw = lw_mvnormal(self.y_obs, self.C_obs, self.y_ref, 
                              cond_thresh=self.cond_thresh)

    def test_values(self):
        # by comparison with scipy implementation
        n = 0
        for m in range(self.M):
            lpd = self.lw[m, n]
            try:
                # scipy implementation raises exception when not pos semidef
                lpd2 = multivariate_normal.logpdf(
                    self.y_obs[n, :], self.y_ref[m, :], self.C_obs[m, ...], 
                    allow_singular=True)
            except:
                lpd2 = None
            if lpd2 is not None:
                self.assertLess(np.abs(lpd-lpd2), 1e-6)
            else:
                self.assertTrue(np.isnan(lpd))
    
    def test_shape(self):
        self.assertEqual(self.lw.shape, (self.M, self.N))
                
                
class MVNormalTestCase2(MVNormalTestCase1):
     
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.__set_C_obs()
     
    @classmethod
    def __set_C_obs(cls):
        C0 = np.diag(np.arange(cls.P) + 3) + 0.5 * np.ones((cls.P, cls.P))
        cls.C_obs = np.stack([C0] * cls.M, axis=0)
        cls.C_obs[1, 0, 0] = 2
        cls.C_obs[2, 0, 0] = 0

if __name__ == '__main__':
    unittest.main()
