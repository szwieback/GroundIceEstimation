'''
Created on Nov 4, 2021

@author: simon
'''
import os
import numpy as np

from analysis import StefanPredictor, PredictionEnsemble
from forcing import read_daily_noaa_forcing, preprocess
from simulation import (StefanStratigraphySmoothingSpline, StratigraphyMultiple)

from scripts.pathnames import paths

def inversion_point():
    N = 25000  # 0
    Nbatch = 10
    geom = {'ia': 30 * np.pi / 180}

    fnforcing = os.path.join(paths['forcing'], 'sagwon2021/daily.csv')
    df = read_daily_noaa_forcing(fnforcing)

    path_ts = os.path.join(paths['processed'], 'dalton/timeseries/2021')
    rad_to_m = -0.05 / (4 * np.pi)  # also change sign convention
    C_obs = np.load(os.path.join(path_ts, 'C_p.npy')) * rad_to_m ** 2
    s_obs = np.load(os.path.join(path_ts, 'unw_p.npy')) * rad_to_m

    d0, d1 = '20210602', '20210910'
    dates = ['20210607', '20210619', '20210701', '20210713', '20210725', '20210806',
             '20210818', '20210830']

    dailytemp, ind_scenes = preprocess(df, d0, d1, dates)

    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N), Nbatch=Nbatch)
    predictor = StefanPredictor()
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)
    predens.extract_predictions(ind_scenes)
    s_pred = predens.extract_predictions(ind_scenes, C_obs=None)

    from inference import lw_mvnormal, psislw, expectation, quantile, _normalize
    lw = lw_mvnormal(s_obs[np.newaxis, :], C_obs[np.newaxis, :], s_pred)
    lw_ps, _ = psislw(lw)
    lw_ = _normalize(lw_ps)
    e_mean = expectation(predens.results['e'], lw_, normalize=False)
    yf_mean = expectation(predens.results['yf'], lw_, normalize=False)
    e_quantile = quantile(
        predens.results['e'], lw_, [0.1, 0.9], normalize=False, smooth=2)
    w_ = np.exp(lw_) * np.ones((predens.ygrid.shape[0], lw_.shape[0]))
    np.putmask(
        w_, predens.ygrid[:, np.newaxis] > predens.results['yf'][..., ind_scenes[-1]], 0)
    print(w_.shape)
    alpha = np.sum(w_, axis=1)

    import matplotlib.pyplot as plt
    from scripts.plotting import cols, prepare_figure
    fig, ax = prepare_figure()
    ax.plot(e_quantile[0, :, 0], predens.ygrid, lw=0.1)
    ax.plot(e_quantile[0, :, 1], predens.ygrid, lw=0.1)
    print(alpha.shape)
    for jdepth in np.arange(predens.ygrid.shape[0] - 1):
            ax.plot(
                e_mean[0, jdepth:jdepth + 2], predens.ygrid[jdepth:jdepth + 2], lw=1.0,
                c=cols['est'], alpha=alpha[jdepth])
            ax.plot(
                e_quantile[0, jdepth:jdepth + 2, 0], predens.ygrid[jdepth:jdepth + 2],
                lw=0.5, c=cols['unc'], alpha=alpha[jdepth])
            ax.plot(
                e_quantile[0, jdepth:jdepth + 2, 1], predens.ygrid[jdepth:jdepth + 2],
                lw=0.5, c=cols['unc'], alpha=alpha[jdepth])

    ax.set_ylim((1.2 * yf_mean[0, ind_scenes[-1]], 0))
    plt.show()

if __name__ == '__main__':
    inversion_point()
