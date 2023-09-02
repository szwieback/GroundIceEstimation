# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

path0 = '/home/simon/Work/gie/ancillary/GEE/'
y_variable = 'ebar'

df = pd.read_csv(os.path.join(path0, 'sample.csv'), index_col=0)
df.dropna(subset=[y_variable], inplace=True)

def xy_split(df, y_variable=y_variable, train_test=True, test_size=0.1, random_state=9999):
    X = pd.get_dummies(df.drop(y_variable, axis=1))
    y = df[y_variable]
    if not train_test:
        return X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
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
                        "Mean validation score: {0:.3f}; std: {1:.3f}".format(
                            results["mean_test_score"][candidate], results["std_test_score"][candidate])
                    )
                    print("Parameters: {0}".format(results["params"][candidate]))
                    print("")

        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer, mean_squared_error
        n_estimators = [100, 300]
        max_features = [0.3, 0.6]
        max_depth = [10, 20, 50]
        max_samples = [0.1] # keep low to deal with overfitting
        param_grid = {'n_estimators':n_estimators,
                      'max_features':max_features,
                      'max_samples': max_samples,
                      'max_depth': max_depth}
        rfr = RandomForestRegressor(oob_score=True)
        scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
        grid = GridSearchCV(rfr, param_grid, scoring=scorer, n_jobs=6, refit=True)
        grid.fit(X_train, y_train)
        report(grid.cv_results_)
        rfr = grid.best_estimator_
    else:
        # rfr = RandomForestRegressor(n_estimators=300, max_features=5, bootstrap=True, oob_score=True)
        rfr = RandomForestRegressor(
            n_estimators=100, max_features=0.3, bootstrap=True, oob_score=True, min_samples_split=10)
        rfr.fit(X_train, y_train)
    predictions = rfr.predict(X_test)
    predictions_train = rfr.predict(X_train)

    # Calculate RMSE on test sample
    from sklearn.metrics import mean_squared_error
    RMSE = mean_squared_error(y_test, predictions, squared=False)
    RMSE_train = mean_squared_error(y_train, predictions_train, squared=False)
    print(f'RMSE = {np.round(RMSE,3)}')
    print(f'RMSE (train) = {np.round(RMSE_train,3)}')
    # # Retrain on entire sample
    # X, y = xy_split(df, train_test=False)
    # rfr.fit(X, y)
    # Save the RF regressor
    if fnout is not None:
        joblib.dump(rfr, fnout)

def X_pred(df, variables, fixed_values=None, N=64, percentiles=(5, 95)):
    if fixed_values is None:
        fixed_values = {'northerliness': 0.0, 'easterliness': 1.0}

    covariates = [key for key in df.columns if key != y_variable]
    X = np.zeros((N * N, len(covariates)))
    consts = [key for key in covariates if (key not in variables)]

    gridvar = [np.linspace(*np.nanpercentile(df[var], percentiles), num=N) for var in variables]
    gridvar_m = np.array(np.meshgrid(*gridvar)).reshape(2, N * N)
    for jvar, var in enumerate(variables):
        X[:, covariates.index(var)] = gridvar_m[jvar,:]
    for _const in consts:
        val = np.nanmedian(df[_const])
        if _const in fixed_values:
            val = fixed_values[_const]
        X[:, covariates.index(_const)] = val
    return gridvar, pd.DataFrame(data=X, columns=covariates)

def predict_grid(df, rfr, variables, fixed_values=None, N=64, percentiles=(1, 99)):
    gridvar, X_p = X_pred(df, variables, fixed_values=fixed_values, N=N, percentiles=percentiles)
    y_p = np.reshape(rfr.predict(X_p), (len(gridvar[0]), len(gridvar[1])))
    return gridvar, y_p

def predict_image(fnim, rfr, fnout):
    # make prediction over entire image and plot
    from analysis import Geospatial
    import rasterio
    import warnings
    from rasterio.windows import Window
    geospatial = Geospatial.from_file(fnim)
    print(geospatial)
    with rasterio.open(fnim) as src:
        shape = (src.height, src.width)
        with rasterio.open(
            fnout, 'w', driver='GTiff', width=shape[1], height=shape[0], count=1,
            dtype=np.float32, crs=src.crs, transform=src.transform) as dst:
            for row in range(src.height):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    window = Window(0, row, shape[1], 1)
                    X = np.moveaxis(src.read(None, window=window), 0, -1)[0, ...]
                    invalid = np.any(np.isnan(X), axis=1)
                    X[invalid,:] = 0.0
                    y = rfr.predict(X)
                    y[invalid] = np.nan
                    dst.write(y[np.newaxis, np.newaxis, ...], window=window)

if __name__ == '__main__':
    fnrf = os.path.join(path0, 'rfr.joblib')
    fnimp = os.path.join(path0, 'importance.joblib')

    run_RF(df, search=True, fnout=fnrf)
    variable_importance(df, fnrf, fnimp)

    fnim = os.path.join(path0, 'covariates.tif')
    fnout = os.path.join(path0, 'e_pred.tif')
    rfr = joblib.load(fnrf)
    predict_image(fnim, rfr, fnout)
