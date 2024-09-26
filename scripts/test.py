from scripts.synthetic_simulation import sagwon_simulation
from scripts.pathnames import paths

replicates = 3
N = 8192
Nsim = 2
Nbatch = 1

sagwon_simulation(
    f'spline_test', N=N, Nsim=Nsim, Nbatch=Nbatch,
    replicates=replicates)