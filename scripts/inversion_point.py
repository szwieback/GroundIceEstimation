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

def inversion_point(site='icecut', fnplot=None):
    N = 50000  # 0
    Nbatch = 1
    geom = {'ia': 30 * np.pi / 180}

    fnforcing = os.path.join(paths['forcing'], 'sagwon2021/daily.csv')
    df = read_daily_noaa_forcing(fnforcing)

    path_ts = os.path.join(paths['processed'], 'dalton/timeseries/2021')
    rad_to_m = -0.055 / (4 * np.pi)  # also change sign convention
    C_obs = np.load(os.path.join(path_ts, f'C_p_{site}.npy')) * rad_to_m ** 2
    C_obs += (0.5e-3) ** 2 * np.eye(C_obs.shape[0])  # add white noise
    s_obs = np.load(os.path.join(path_ts, f'unw_p_{site}.npy')) * rad_to_m
    print(np.load(os.path.join(path_ts, f'unw_p_{site}.npy')))
    print(s_obs)
    print(np.sqrt(np.diag(C_obs)))

    d0, d1 = '20210602', '20210910'
#     dates = ['20210607', '20210619', '20210701', '20210713', '20210725', '20210806',
#              '20210818', '20210830']
    dates = np.load(os.path.join(path_ts, 'scenes.npy'))
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
    s_obs_ = np.concatenate(([0], s_obs))
    s_mean = expectation(predens.results['s_los'], lw_, normalize=False)
    e_quantile = quantile(
        predens.results['e'], lw_, [0.1, 0.9], normalize=False, smooth=2)
    w_ = np.exp(lw_) * np.ones((predens.ygrid.shape[0], lw_.shape[0]))
    np.putmask(
        w_, predens.ygrid[:, np.newaxis] > predens.results['yf'][..., ind_scenes[-1]], 0)
    alpha = np.sum(w_, axis=1)

    if fnplot is not None:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from scripts.plotting import cols, prepare_figure
        fig, ax = prepare_figure(
            ncols=2, figsize=(1.0, 0.5), sharex=False, sharey=False, wspace=0.4, right=0.98, 
            bottom=0.22, left=0.16)
        ax[0].plot(
            dailytemp.index, s_mean[0, :] - s_mean[0, ind_scenes[0]], c=cols['est'], lw=1.0)
        ax[0].plot(dailytemp.index[ind_scenes], s_obs_,  lw=0.0, c='k',
            alpha=0.6, marker='o', mfc='k', mec='none', ms=4)
        for jdepth in np.arange(predens.ygrid.shape[0] - 1):
                ax[1].plot(
                    e_mean[0, jdepth:jdepth + 2], predens.ygrid[jdepth:jdepth + 2], lw=1.0,
                    c=cols['est'], alpha=alpha[jdepth])
                ax[1].plot(
                    e_quantile[0, jdepth:jdepth + 2, 0], predens.ygrid[jdepth:jdepth + 2],
                    lw=0.5, c=cols['unc'], alpha=alpha[jdepth])
                ax[1].plot(
                    e_quantile[0, jdepth:jdepth + 2, 1], predens.ygrid[jdepth:jdepth + 2],
                    lw=0.5, c=cols['unc'], alpha=alpha[jdepth])
        ax[0].set_ylim((np.max(s_obs) * 1.2, -0.002))
        ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax[0].text(
            -0.34, 0.50, 'subsidence [m]', ha='right', va='center', rotation=90, 
            transform=ax[0].transAxes)
        ax[1].text(
            -0.28, 0.50, 'depth [m]', ha='right', va='center', rotation=90, 
            transform=ax[1].transAxes)
        ypos=-0.28
        ax[0].text(
            0.5, ypos, d0[0:4], ha='center', va='baseline', transform=ax[0].transAxes)
        ax[0].text(
            0.5, ypos, 'excess ice [-]', ha='center', va='baseline', 
            transform=ax[1].transAxes)
        ax[1].set_ylim((1.2 * yf_mean[0, ind_scenes[-1]], 0))
        plt.savefig(fnplot)

if __name__ == '__main__':
    for site in ['happyvalley']:#, 'icecut'
        fnplot = os.path.join(paths['figures'], f'{site}_2021.pdf')
        inversion_point(site=site, fnplot=fnplot)
