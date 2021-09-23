'''
Created on Sep 23, 2021

@author: simon
'''
import numpy as np

from inference import lw_mvnormal, psislw, ensemble_quantile

rng = np.random.default_rng(seed=1)
A = np.array([[1, 1, 0], [0, 1, 1]])
C_obs = 3e-4 * np.eye(A.shape[0]) # the smaller C_obs, the bigger the coverage problems

Ne = 20000
Ns = 500
Nr = 5

theta_ens = rng.multivariate_normal(np.zeros(A.shape[1]), np.eye(A.shape[1]), size=(Ne,))
X_ens = np.einsum('ijk, ik -> ij', A[np.newaxis, ...], theta_ens)

theta_sim = rng.multivariate_normal(np.zeros(A.shape[1]), np.eye(A.shape[1]), size=(Ns,))
X_sim0 = np.einsum('ijk, ik -> ij', A[np.newaxis, ...], theta_sim)
noise = rng.multivariate_normal(np.zeros(A.shape[0]), C_obs, size=(Nr,))
X_sim = X_sim0[np.newaxis, ...] + noise[:, np.newaxis, ...]

eq = []
for jr in range(Nr):
    _C_obs = np.broadcast_to(C_obs, (X_sim.shape[1],) + C_obs.shape)
    lwr = lw_mvnormal(X_sim[jr, ...], _C_obs, X_ens)
    lwr_ps, _ = psislw(lwr)
#     lw.append(lwr_ps)
    eqr = ensemble_quantile(theta_ens, lwr_ps, theta_sim)
    eq.append(eqr)
eq = np.array(eq)

target = 0.8
covbool = np.abs(eq - 0.5) < target / 2
coverage = np.mean(covbool, axis=0)
print(np.mean(coverage, axis=0))

# print(X_sim)
#
# Nr = 1000
#
# obs_noise = rng.multivariate_normal(
#     np.zeros(C_obs.shape[0]), C_obs, size=(Nr,))
# X_obs = X_sim[np.newaxis, :] + obs_noise
# print(X_obs.shape)
