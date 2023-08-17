# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

path0 = '/home/simon/Work/gie/ancillary/GEE/'
y_variable = 'ebar'
def xy_split(df, y_variable=y_variable, train_test=True):
    X = pd.get_dummies(df.drop(y_variable, axis=1))
    y = df[y_variable]
    if not train_test:
        return X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
        return X_train, X_test, y_train, y_test 

def variable_importance(df, fnrf, fnimp):
    rfr = joblib.load(fnrf)
    names = list(df.columns)[1:]
    # standard impurity based importance
    fi = rfr.feature_importances_
    X, y = xy_split(df, train_test=False)
    # compute impurity importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(rfr, X, y, n_repeats=5, random_state=0)
    fip = result.importances_mean
    impres = {'names': names, 'permutation': fip, 'standard': fi}
    joblib.dump(impres, fnimp)
    
def run_RF(df, search=False, fnout=None):
    X_train, X_test, y_train, y_test = xy_split(df, train_test=True)
    if search:
        # Grid search for hyperparameter tuning
        def report(results, n_top=3):
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results["rank_test_score"] == i)
                for candidate in candidates:
                    print("Model with rank: {0}".format(i))
                    print(
                        "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                            results["mean_test_score"][candidate],
                            results["std_test_score"][candidate],
                        )
                    )
                    print("Parameters: {0}".format(results["params"][candidate]))
                    print("")
    
        from sklearn.model_selection import GridSearchCV
        n_estimators = [200, 300, 400]
        max_features = [3, 4, 5]
        bootstrap = [True]
        param_grid = {'n_estimators':n_estimators,
                     'max_features':max_features,
                     'bootstrap':bootstrap}
        rfr = RandomForestRegressor()
        grid = GridSearchCV(rfr, param_grid)
        grid.fit(X_train, y_train)
        report(grid.cv_results_)
    
    # Fit Random Forest Regressor with tuned hyperparameters
    rfr = RandomForestRegressor(n_estimators=300, max_features=5, bootstrap=True, oob_score=True)
    rfr.fit(X_train, y_train)
    predictions = rfr.predict(X_test)
    
    # Calculate RMSE on test sample
    from sklearn.metrics import mean_squared_error
    RMSE = mean_squared_error(y_test, predictions, squared=False)  # If True returns MSE value, if False returns RMSE value.
    print(f'RMSE = {np.round(RMSE,3)}')
    
    # Retrain on entire sample
    X, y = xy_split(df, train_test=False)
    rfr.fit(X, y)
    # Save the RF regressor
    if fnout is not None:
        joblib.dump(rfr, fnout)

def plot_fit(df, rfr, impres):
    import matplotlib.pyplot as plt
    from scripts.plotting import prepare_figure
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
    
def X_pred(df, variables, fixed_values=None, N=64, percentiles=(5, 95)):
    if fixed_values is None:
        fixed_values = {'northerliness': 0.0, 'easterliness': 1.0}

    covariates = [key for key in df.columns if key != y_variable]
    X = np.zeros((N*N, len(covariates)))
    consts = [key for key in covariates if (key not in variables)]

    gridvar = [np.linspace(*np.nanpercentile(df[var], percentiles), num=N) for var in variables]
    gridvar_m = np.array(np.meshgrid(*gridvar)).reshape(2, N*N)
    for jvar, var in enumerate(variables):
        X[:, covariates.index(var)]= gridvar_m[jvar, :]
    for _const in consts:
        val = np.nanmedian(df[_const])
        if _const in fixed_values:
            val = fixed_values[_const]
        X[:, covariates.index(_const)] = val  
    return gridvar, pd.DataFrame(data=X, columns=covariates)
    
def predict(df, rfr, variables, fixed_values=None, N=64, percentiles=(5, 95)):
    gridvar, X_p = X_pred(df, variables, fixed_values=fixed_values, N=N, percentiles=percentiles)
    y_p = np.reshape(rfr.predict(X_p), (len(gridvar[0]), len(gridvar[1])))
    return gridvar, y_p
    
def plot_pred(df, rfr):
    vlim = (0.0, 0.5)
    variables = ('slope', 'NDWI')
    fixed_values = {'northerliness': 0.0, 'easterliness': 1.0, 'DEM_bp': 0}
    fixed_values_plot = [{'ndvi': 0.2}, {'ndvi': 0.5}, {'ndvi': 0.8}]
    
    import matplotlib.pyplot as plt
    from scripts.plotting import prepare_figure, cmap_e
    fig, axs = prepare_figure(
        ncols=3, figsize=(1.6, 0.5), sharex=False, sharey=False, wspace=1, bottom=0.22)
    for jp, _fv in enumerate(fixed_values_plot):
        fv = {**fixed_values, **_fv}
        gridvar, y_p = predict(df, rfr, variables, fixed_values=fv)
        axs[jp].imshow(
            y_p, extent=(gridvar[0][0], gridvar[0][-1], gridvar[1][0], gridvar[1][-1]), aspect='auto',
            vmin=vlim[0], vmax=vlim[1], cmap=cmap_e)
        axs[jp].set_xlabel(variables[0])
        axs[jp].set_ylabel(variables[1])
    plt.show()
        
    
    
    
    # use meshgrid to create predictions
    # pass
    # plot predicted versus true; variable importance 
    #slope vs. ndwi for average NDVI, etc. plot for two slopes: NDVI, NDWI for average ruggedness, elevation, 
    # all east-facing



if __name__ == '__main__':
    df = pd.read_csv(os.path.join(path0, "sample.csv"), index_col=0)
    df.dropna(subset=['ebar'], inplace=True)
    fnrf = os.path.join(path0, 'rfr.joblib')
    fnimp = os.path.join(path0, 'importance.joblib')
    
    # run_RF(df, fnout=fnrf)
    # variable_importance(df, fnrf, fnimp)

    rfr = joblib.load(fnrf)
    impres = joblib.load(fnimp)
    # plot_fit(df, rfr, impres)
    plot_pred(df, rfr)
