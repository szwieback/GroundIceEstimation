'''
Created on Jul 23, 2020

@author: simon
'''
import os
import pandas as pd
import datetime
import numpy as np
from simulation.stefan import stefan_ens, stefan_integral_balance
from simulation.stratigraphy import StefanStratigraphy, StefanStratigraphySmoothingSpline

def read_merra_subset(fn, field='T2MMEAN[0][0]'):
    with open(fn, 'r') as f:
        dstr = f.readline().split('.')[-2]
        d = datetime.datetime.strptime(dstr, '%Y%m%d')
        parts = [p.strip() for p in f.readline().split(', ')]
        assert len(parts) == 2
        assert parts[0] == field
        T = float(parts[1])
    return d, T

def load_forcing_merra_subset(folder, to_Celsius=True):
    ld = os.listdir(folder)
    def _match(fn1, fn2, comps=(0, 1, 3, 4)):
        p1, p2 = fn1.split('.'), fn2.split('.')
        return all([p1[co] == p2[co] for co in comps])
    fn0 = ld[0]
    fns = [fn for fn in ld if _match(fn, fn0)]
    vals = [read_merra_subset(os.path.join(folder, fn)) for fn in fns]
    df = pd.DataFrame(vals, columns=('datetime', 'T'))
    df.sort_values(by='datetime', inplace=True)
    df = df.set_index('datetime')
    if to_Celsius:
        df['T'] = df['T'] - 273.15
    return df

def get_displacements(year=2019, locn='ridge'):
    from InterferometricSpeckle.storage import load_object, save_object, prepare_figure
    from pathnames import paths
    stackname = 'kivalina' + str(year)
    pathout = os.path.join(paths['processed'], stackname, 'timeseries')
    datesstr = {'kivalina2019':
             ('20190606', '20190618', '20190630', '20190712', '20190724', '20190805',
              '20190817', '20190829', '20190910'),
             'kivalina2017':
             ('20170604', '20170616', '20170628', '20170710', '20170722', '20170803', '20170815',
              '20170827', '20170908', '20170920')}
    dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[stackname]]
    fn = os.path.join(pathout, f'disp_{locn}.p')
    dispd = load_object(fn)
    return dates, dispd

def parse_dates(datestr, strp='%Y%m%d'):
    if isinstance(datestr, str):
        return datetime.datetime.strptime(datestr, strp)
    else:
        return [parse_dates(ds, strp=strp) for ds in datestr]

if __name__ == '__main__':
    folder = '/home/simon/Work/gie/forcing/Kivalina'
    df = load_forcing_merra_subset(folder)
    year = 2019
    d0 = {2019: '2019-05-10', 2017: '2017-05-10', 2018: '2018-05-10'}[year]
    d1 = {2019: '2019-09-15', 2017: '2017-09-20', 2018: '2018-09-15'}[year]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())['T'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0

    datesdisp, dispd = get_displacements(year=year)

    params_dist = {
        'Nb': 8, 'expb': 1.3, 'b0': 0.03, 'bm': 0.65,
        'e': {'alpha_shape': 1.0, 'beta_shape': 1.5, 'high_scale': 0.9, 'alpha_shift': 0.1,
              'beta_shift': 1.0},
        'wsat': {'low_above': 0.4, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
        'soil': {'high_horizon': 0.2, 'low_horizon': 0.05, 'organic_above': 0.1,
                 'mineral_above': 0.1, 'mineral_below': 0.3, 'organic_below': 0.05},
        'n_factor': {'high': 0.95, 'low': 0.85, 'alphabeta': 2.0}}

    params_dist = {
        'Nb': 10, 'expb': 1.5, 'b0': 0.05, 'bm': 0.70,
        'e': {
            'low': 0.00, 'high': 0.95, 'coeff_mean': 2, 'coeff_std': 3, 'coeff_corr': 0.7},
        'wsat': {'low_above': 0.3, 'high_above': 0.9, 'low_below': 0.8, 'high_below': 1.0},
        'soil': {'high_horizon': 0.3, 'low_horizon': 0.1, 'organic_above': 0.1,
                 'mineral_above': 0.05, 'mineral_below': 0.3, 'organic_below': 0.05},
        'n_factor': {'high': 0.95, 'low': 0.85, 'alphabeta': 2.0}}

    strat = StefanStratigraphySmoothingSpline(dist=params_dist, N=30000)
    strat.draw_stratigraphy()
    params = strat.params()
    print(np.mean(params['e'][:, 200]))
    print(strat._cpoints())
    dailytemp_ens = dailytemp

    s, yf = stefan_integral_balance(dailytemp_ens, params=params, steps=1)
    ind_obs = [int((d - d0_).days) for d in datesdisp][1:]
    s_obs_pred = s[:, ind_obs]

    s_obs = -dispd['disp'][np.newaxis, :] 
    C = dispd['C'][np.newaxis, ...] # scaling has a noticeable eff
    C = (np.eye(len(s_obs[0, ...])) * 0.01 ** 2)[np.newaxis, ...]
    from inference import psislw, lw_mvnormal, expectation
    lw = lw_mvnormal(s_obs, C, s_obs_pred)
    lw_ps, _ = psislw(lw)

    e_est = expectation(params['e'], lw_ps, normalize=True)
    s_obs_pred_est = expectation(s_obs_pred, lw_ps, normalize=True)
    s_est = expectation(s, lw_ps, normalize=True)
    yf_est = expectation(yf, lw_ps, normalize=True)
    print(yf_est[0, -1])
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=2, sharey=False)
    fig.set_size_inches((8, 3), forward=True)
    days = np.arange(len(dailytemp))

    axs[0].plot(days[ind_obs], s_obs[0, :], lw=0.0, c='k', alpha=0.6, marker='o',
            mfc='k', mec='none', ms=4)
    axs[0].plot(days, s_est[0, :], lw=1.0, c='#999999', alpha=0.6)
#     axs[0].plot(days, s[ind_large[0, -1], :], lw=0.5, c='#ccccff', alpha=0.5)
#     axs[0].plot(days, s[ind_large[0, -2], :], lw=0.5, c='#ccccff', alpha=0.5)
#     axs[0].plot(days, s[ind_large[0, 9000], :], lw=0.5, c='#ffcccc', alpha=0.5)
    axs[1].plot(e_est[0, :], strat._ygrid, lw=1.0, c='#999999', alpha=0.6)
    plt.show()

