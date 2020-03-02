'''
Created on Mar 1, 2020

@author: simon
'''
import unittest
import numpy as np

from inference.gmi import posterior_gm_mvnormal, fit_gaussian_mixture

class PosteriorGMMvNormalTestDiagonal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Q = 20
        cls.P = 9
        cls.N = 1000
        cls.M = 3
        cls.K = 4
        Sigma0 = np.eye(cls.Q)
        C_obs_std = np.diag(np.arange(cls.P) + 1)
        cls.C_obs = np.stack([C_obs_std] * cls.M, axis=0)
        cls.C_obs[1, 0, 0] = 1e-9
        sigma2s = (1.0, 3.0)
        pis = (0.3, 0.7)
        
        mus = (np.zeros(cls.Q), 2 * np.ones(cls.Q))
        rs = np.random.RandomState(seed=1)
        samples = []
        for jcomp in np.arange(len(sigma2s)):
            samples.append(
                rs.multivariate_normal(mean=mus[jcomp], cov=sigma2s[jcomp] * Sigma0,
                                       size=(int(pis[jcomp] * cls.N),)))
        samples = np.concatenate(samples, axis=0)
        cls.gm = fit_gaussian_mixture(samples, random_state=rs, n_components=cls.K,
                                       n_init=2)
        cls.y_obs = np.ones((cls.M, cls.P))

    def setUp(self):
        self.mu_p, self.Sigma_p, self.pi_p = posterior_gm_mvnormal(
            self.y_obs, self.C_obs, self.gm, H=None, cond_thresh=1e-6, 
            method_condition='square_root')

    def test_methods(self):
        mu_p2, Sigma_p2, pi_p2 = posterior_gm_mvnormal(
            self.y_obs, self.C_obs, self.gm, H=None, cond_thresh=1e-6, 
            method_condition='full')
        self.assertLess(np.nanmean(np.abs(mu_p2 - self.mu_p)), 1e-6)
        self.assertLess(np.nanmean(np.abs(Sigma_p2 - self.Sigma_p)), 1e-6)
        self.assertLess(np.nanmean(np.abs(pi_p2 - self.pi_p)), 1e-6)
        
    def test_shape(self):
        self.assertEqual(self.mu_p.shape, (self.M, self.K, self.Q))
        self.assertEqual(self.Sigma_p.shape, (self.M, self.K, self.Q, self.Q))
        self.assertEqual(self.pi_p.shape, (self.M, self.K))

    def test_sum(self):
        self.assertLess(np.nanmean(np.abs(1 - np.nansum(self.pi_p, axis=1))), 1e-6)

class PosteriorGMMvNormalTestInvalid(PosteriorGMMvNormalTestDiagonal):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        C_obs_std = np.diag(np.arange(cls.P) + 1) + 0.2 * np.ones((cls.P, cls.P))
        cls.C_obs = np.stack([C_obs_std] * cls.M, axis=0)
        cls.C_obs[1, 0, 0] = 1e-9
        cls.C_obs[2, 0, 0] = -1

    def test_methods(self):
        pass
    
    def test_sum(self):
        pass
    
    def test_nan(self):
        self.assertFalse(np.any(np.isfinite(self.pi_p[1, :])))
        self.assertFalse(np.any(np.isfinite(self.pi_p[2, :])))
        self.assertFalse(np.any(np.isfinite(self.mu_p[1, :])))
        self.assertFalse(np.any(np.isfinite(self.mu_p[2, :])))
        self.assertFalse(np.any(np.isfinite(self.Sigma_p[1, :])))
        self.assertFalse(np.any(np.isfinite(self.Sigma_p[2, :])))

if __name__ == '__main__':
    unittest.main()
