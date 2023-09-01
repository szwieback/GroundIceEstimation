'''
Created on Aug 29, 2023

@author: simon
'''
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

from scripts.kivalina_RF import xy_split, predict_grid, path0, df
from scripts.plotting import prepare_figure, colslist, cmap_e
from scripts.pathnames import paths

cmap_e_clipped = colors.LinearSegmentedColormap.from_list('clipped', cmap_e(np.linspace(0.0, 0.7, 256)))

def plot_fit(df, rfr, impres):
    import matplotlib.pyplot as plt
    fig, axs = prepare_figure(ncols=2, figsize=(1.0, 0.4), sharex=False, sharey=False, wspace=1)
    X, y = xy_split(df, train_test=False)
    axs[0].plot(y, rfr.predict(X), linestyle='none', mec='none', mfc='k', ms=1, marker='o', alpha=0.1)
    fitype = 'permutation'
    ind = np.argsort(impres[fitype])
    print(impres[fitype])
    axs[1].plot(impres[fitype][ind], np.arange(len(ind)), linestyle='none', marker='o')
    axs[1].set_yticks(np.arange(len(ind)))
    axs[1].set_yticklabels(np.array(df.columns)[1:][ind])
    axs[1].set_xlim((0.0, 0.7))
    plt.show()

def plot_pred(df, rfr):
    vlim = (0.0, 0.5)
    variables = ('slope', 'NDWI')
    fixed_values = {'northerliness': 0.0, 'easterliness': 1.0, 'DEM_bp': 0, 'rugged': 30}
    fixed_values_plot = [{'ndvi': 0.25}, {'ndvi': 0.75}]
    # dem_bp also has big influence, but RF did not learn rocky (DEM_bp >>0,NDVI~0, NDWI<0) well
    import matplotlib.pyplot as plt
    from scripts.plotting import prepare_figure, cmap_e
    fig, axs = prepare_figure(
        ncols=len(fixed_values_plot), figsize=(1.6, 0.5), sharex=False, sharey=False, wspace=1, bottom=0.22)
    for jp, _fv in enumerate(fixed_values_plot):
        fv = {**fixed_values, **_fv}
        gridvar, y_p = predict_grid(df, rfr, variables, fixed_values=fv)
        print(gridvar[1])
        axs[jp].imshow(
            y_p, extent=(gridvar[0][0], gridvar[0][-1], gridvar[1][0], gridvar[1][-1]), aspect='auto',
            vmin=vlim[0], vmax=vlim[1], cmap=cmap_e, origin='lower')
        axs[jp].set_xlabel(variables[0])
        axs[jp].set_ylabel(variables[1])
    plt.show()

def conditional_expectation(
        rfr, X, covariate, ranges_dict=None, covariate_range=None, covariate_steps=64, subsample=10, 
        rng=999):
    # single covariate
    try:
        rng.random()
    except:
        rng = np.random.default_rng(rng)
    if covariate_range is None:
        covariate_range = np.nanpercentile(X[covariate], (1, 99))
    covariate_grid = np.linspace(*covariate_range, covariate_steps)
    _X = X
    if ranges_dict is not None:
        for var in ranges_dict:
            if ranges_dict[var][0] is not None:
                _X = _X[_X[var] > ranges_dict[var][0]]
            if ranges_dict[var][1] is not None:
                _X = _X[_X[var] <= ranges_dict[var][1]]
    if subsample is not None and len(_X) > subsample:
        _X = _X.iloc[rng.permutation(len(_X))[0:subsample]]
    y = []
    cyp= []
    for ind in range(len(_X)):
        df = pd.DataFrame(_X.iloc[ind:ind + 1])
        dfr = df.loc[np.repeat(df.index.values, covariate_steps)]
        dfr[covariate] = covariate_grid
        y_ind = rfr.predict(dfr)
        yp_ind = rfr.predict(df)
        cyp.append([df[covariate].iloc[0], yp_ind[0]])
        y.append(y_ind)
    dict_res = {'X': _X, 'y_pred_cov': np.array(y).T, 'cov_y_pred': np.array(cyp), 'grid': covariate_grid,
                'covariate': covariate, 'ranges_dict': ranges_dict}
    return dict_res

def plot_conditional_expectation(
        ce_res, ax, cmap=None, clim=(0, 1), ccov=None, alpha=0.5, markersize=2.5, mew=0.5, lw=1.0, marker='o'):
    if cmap is None:
        import matplotlib
        cmap = matplotlib.colormaps['viridis']
    grid = ce_res['grid']
    if ccov is not None:
        cvals = (np.array(ce_res['X'][ccov]) - clim[0]) / (clim[1] - clim[0])
        c = cmap(cvals)
    else:
        c = cmap(np.ones_like(ce_res['y_pred_cov']))
    for _y, _cy, _c in zip(ce_res['y_pred_cov'].T, ce_res['cov_y_pred'], c):
        ax.plot(grid, _y, alpha=alpha, c=_c, lw=lw, zorder=3)
        if markersize > 0:
            ax.plot(
                _cy[0], _cy[1], c=_c, alpha=1.0, markersize=markersize, marker=marker, 
                markeredgecolor='none', linestyle='none', mew=mew, zorder=4)
            ax.plot(
                _cy[0], _cy[1], markersize=markersize, alpha=0.8, marker=marker, markeredgecolor='#ffffff', 
                markerfacecolor='none', linestyle='none', mew=mew, zorder=5)            

def plot_rf(df, rfr, impres, fnout=None):
    from sklearn.metrics import mean_squared_error
    from matplotlib.transforms import Bbox
    from matplotlib.colors import Normalize
    from matplotlib import cm
    from string import ascii_lowercase
    fig, axs = prepare_figure(
        ncols=5, figsize=(2.03, 0.39), sharex=False, sharey=False, wspace=0.670, bottom=0.178, top=0.968,
        right=0.990, left=0.100)
    dx = -0.03
    cbarl = 0.94
    X_train, X, y_train, y = xy_split(df, train_test=True)
    axs[1].plot((0, 1), (0, 1), lw=0.5, alpha=0.3, c=colslist[1])
    pred = rfr.predict(X)
    axs[1].plot(
        y, pred, linestyle='none', mec='none', mfc=colslist[0], ms=1, marker='o', alpha=0.1)
    RMSE = mean_squared_error(y, pred, squared=False)
    axs[1].text(0.04, 0.90, f'RMSE {RMSE:1.2f}', transform=axs[1].transAxes, ha='left')
    lims = (0.00, 0.65)
    ticks = (0.0, 0.3, 0.6)
    pos_xl = (0.50, -0.41)
    pos_yl = (-0.36, 0.45)
    axs[1].set_xlim(lims)
    axs[1].set_ylim(lims)
    axs[1].set_yticks(ticks)
    axs[1].text(
        *pos_xl, '$\\hat{\\bar{e}}$ [$-$]', ha='center', va='baseline', transform=axs[1].transAxes)
    axs[1].text(
        *pos_yl, 'RF $\\bar{e}$ [$-$]', ha='right', va='center', transform=axs[1].transAxes, rotation=90)
    fitype = 'permutation'
    ind = np.argsort(impres[fitype])
    # print(impres[fitype])
    axs[0].plot(
        impres[fitype][ind], np.arange(len(ind)), linestyle='none', marker='o', ms=2, mec='none',
        mfc=colslist[0])
    covariate_labels = {
        'ndvi': 'NDVI', 'NDWI': 'NDWI', 'elevation': 'elevation', 'rugged': 'ruggedness',
        'DEM_bp': 'elev. bp', 'slope': 'slope', 'northerliness': 'north', 'easterliness': 'east'}
    axs[0].set_yticks(np.arange(len(ind)))
    axs[0].set_yticklabels([covariate_labels[x] for x in np.array(df.columns)[1:][ind]])
    axs[0].set_xlim((0.0, 0.7))
    axs[0].text(
        *pos_xl, 'importance [-]', ha='center', va='baseline', transform=axs[0].transAxes)

    # from sklearn.inspection import PartialDependenceDisplay
    # dpd = PartialDependenceDisplay.from_estimator(
    #     rfr, X, ['slope'], kind='individual', ax=axs[2], subsample=100)
    # print(dpd)
    # PartialDependenceDisplay.from_estimator(
    #     rfr, X, ['slope', 'NDWI', 'ndvi'], ax=axs[2], subsample=100)
    # print(np.count_nonzero(X['ndvi'] > 0.73))
    ccov, clim = 'NDWI', (-0.05, 0.65)
    alpha = 0.5
    cmap = cmap_e_clipped
    
    ndvi_ranges = [(None, 0.30), (0.30, 0.75), (0.75, None)]
    subsample = 12
    cov, lims_cov = 'slope', (0.02, 10)
    for jndvir, ndvir in enumerate(ndvi_ranges):
        ce_res = conditional_expectation(
            rfr, X_train, cov, covariate_range=(0.5, 20), ranges_dict={'ndvi': ndvir}, subsample=subsample)
        ax =  axs[2 + jndvir]
        plot_conditional_expectation(
            ce_res, ax, cmap=cmap, clim=clim, ccov=ccov, alpha=alpha)
        if jndvir > 0:
            ax.set_yticklabels([])
        ax.set_ylim(lims)
        ax.set_xlim(lims_cov)
    axs[2].text(
        *pos_yl, 'RF $\\bar{e}$ [$-$]', ha='right', va='center', transform=axs[2].transAxes, rotation=90)
    
    n_shift = 0
    labels = ['importance', 'fit', 'low NDVI', 'medium NDVI', 'high NDVI']
    for jax, ax in enumerate(axs):
        ax.set_aspect(1 / ax.get_data_ratio())
        label = f'{ascii_lowercase[jax]}) {labels[jax]}'
        ax.text(0.00, 1.06, label, ha='left', va='baseline', transform=ax.transAxes)
        if jax in [2, 3, 4]:
            apos = ax.get_position()
            ax.set_yticks(ticks)
            ax.set_position(Bbox.from_bounds(apos.x0 + n_shift * dx, apos.y0, apos.width, apos.height))
            ax.text(
                *pos_xl, 'slope [$^{\\circ}$]', ha='center', va='baseline', transform=ax.transAxes)
            n_shift = n_shift + 1
    dh = 0.08
    cax = fig.add_axes((cbarl, apos.y0 + dh, 0.01, apos.height-2*dh))
    cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(*clim, clip=True), cmap=cmap), cax=cax)
    cbarlabel = 'NDWI [-]'
    cax.text(2.40, 1.07, cbarlabel, ha='center', va='baseline', transform=cax.transAxes)            
    if fnout is not None:
        fig.savefig(fnout)
    else:
        plt.show()

if __name__ == '__main__':
    fnrf = os.path.join(path0, 'rfr.joblib')
    fnimp = os.path.join(path0, 'importance.joblib')
    rfr = joblib.load(fnrf)
    impres = joblib.load(fnimp)

    fnplot = os.path.join(paths['figures'], 'index/rf.pdf')
    plot_rf(df, rfr, impres, fnout=fnplot)

    # plot_fit(df, rfr, impres)

    # plot_pred(df, rfr)
