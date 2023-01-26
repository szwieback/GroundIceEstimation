'''
Created on Sep 14, 2022

@author: simon
'''

import numpy as np
from itertools import product

wvl0 = 0.055

def phase_to_length_factor(wavelength=wvl0):
    # two-way prop
    conv = wavelength / (4 * np.pi)
    return conv

def var_atmo_conversion(var_atmo, to_length=True, wavelength=wvl0, var_atmo_phase=False):
    conv = phase_to_length_factor(wavelength)
    _var_atmo = var_atmo
    if var_atmo_phase and to_length:
        _var_atmo = var_atmo * (conv ** 2)
    elif not to_length and not var_atmo_phase:
        _var_atmo = var_atmo / (conv ** 2)
    return _var_atmo

def add_atmospheric_K(K, var_atmo, wavelength=wvl0, to_length=True, var_atmo_phase=False):
    conv = phase_to_length_factor(wavelength)
    if to_length:
        K *= conv ** 2
    K_eye = np.eye(K.shape[0])[(Ellipsis,) + (None,) * (len(K.shape) - 2)]
    _var_atmo = var_atmo_conversion(
        var_atmo, to_length=to_length, wavelength=wavelength, var_atmo_phase=var_atmo_phase)
    K += _var_atmo * (np.ones_like(K) + K_eye)
    return K

def compute_distances(xy_point, xy_ref, geospatial):
    def _compute_distance_jref(jref):
        _xy_ref = xy_ref[:, jref][:, np.newaxis]
        xyl = np.concatenate((xy_point, _xy_ref), axis=1)
        return geospatial.distance(xyl.T)
    return np.array([_compute_distance_jref(jref) for jref in range(xy_ref.shape[1])])

def distance_to_ref(
        geospatial, xy_ref, fndist=None, geospatial_ref=None, overwrite=False, njobs=-2, block_size=256):
    # slow, but probably not a bottleneck
    import geopandas as gpd
    from shapely.geometry import Point
    gsp_ref = geospatial_ref if geospatial_ref is not None else geospatial
    s_ref = gpd.GeoSeries([Point(xy_ref[:, jref]) for jref in range(xy_ref.shape[1])], crs=gsp_ref.crs)
    crs_dist = s_ref.estimate_utm_crs()
    s_ref_proj = s_ref.to_crs(crs_dist)
    x_ref_proj, y_ref_proj = np.array(s_ref_proj.x), np.array(s_ref_proj.y)
    if fndist is None or overwrite or not os.path.exists(fndist):
        xy_raster = geospatial.xy_raster()
        if njobs not in [0, 1, None]:
            from joblib import Parallel, delayed
            xy_blocks = np.array_split(np.reshape(xy_raster, (-1, 2)), block_size, axis=0)
            def _process_block(xy_block):
                s_block = gpd.GeoSeries(
                    [Point(_xy) for _xy in xy_block], crs=geospatial_K.crs)
                s_block_proj = s_block.to_crs(crs_dist)
                x_block, y_block = np.array(s_block_proj.x), np.array(s_block_proj.y)
                dist = np.stack(
                [(x_block - x) ** 2 + (y_block - y) ** 2 for x, y in zip(x_ref_proj, y_ref_proj)], axis=-1)
                return dist
            dist = np.concatenate(
                Parallel(n_jobs=njobs)(delayed(_process_block)(block) for block in xy_blocks))
        else:
            s_raster = gpd.GeoSeries(
                [Point(_xy) for _xy in np.reshape(xy_raster, (-1, 2))], crs=geospatial_K.crs)
            s_raster_proj = s_raster.to_crs(crs_dist)
            x_raster, y_raster = np.array(s_raster_proj.x), np.array(s_raster_proj.y)
            dist = np.stack(
                [(x_raster - x) ** 2 + (y_raster - y) ** 2 for x, y in zip(x_ref_proj, y_ref_proj)], axis=-1)
        distances = np.reshape(dist, geospatial.shape + (-1,))
    else:
        distances = np.load(fndist)
        _dist = compute_distances(geospatial.xy(np.array((0, 0))[:, np.newaxis]), xy_ref, geospatial)
        np.testing.assert_allclose(distances[:, 0, 0], _dist)  # first pixel does not match
    return distances

def add_atmospheric_K_spatial(
        K, xy_ref, cov_fun_atmos, geospatial, wavelength=wvl0, to_length=True, dist=None):
    shape = K.shape
    assert len(shape) == 4  # 2d image
    assert shape[0] == shape[1]  # covariance matrix
    if dist is None:
        dist = distance_to_ref(geospatial, xy_ref)

        # print(geospatial.distance(xyl.T))
if __name__ == '__main__':
    import os

    path0 = f'/home/simon/Work/gie/processed/kivalina/2019'
    geom = {'ia': 39.29 / 180 * np.pi}
    wavelength = 0.055
    var_atmo = (4e-3) ** 2
    xy_ref = np.array([-164.7300, 67.8586])[:, np.newaxis]
    xy_ref = np.array([[-164.7300, 67.8586], [-164.7350, 67.8500]]).T

    from analysis import (
        read_K, add_atmospheric, read_referenced_motion, InversionProcessor,
        InversionResults, Geospatial)

    fnunw = os.path.join(path0, 'unwrapped.geo.tif')
    fnK = os.path.join(path0, 'K_vec.geo.tif')
    K, geospatial_K = read_K(fnK)
    def cov_fun_atmos(distances):
        outshape = (K.shape[0] + 1,) + (1, ) * len(distances.shape)
        return np.ones(outshape) * distances[np.newaxis, ...] * 1e-6 + 1e-4
    # add_atmospheric_spatial(K, xy_ref, cov_fun, geospatial_K)
    fndist = os.path.join(path0, 'distance.p')
    # dist = distance_to_ref(geospatial_K, xy_ref, fndist=fndist, overwrite=False)
    import copy
    geospatial_test = copy.deepcopy(geospatial_K)
    geospatial_test.shape = (50, 30)
    distances = distance_to_ref(geospatial_K, xy_ref, fndist=fndist, overwrite=False)
    print(distances.shape)
    K_atmos = cov_fun_atmos(distances)
    print(K_atmos.shape)
    print(K.shape)
    
    # var of weighted average across all ref points (weights as input)
    # covariance matrix of difference (use zero as temp. ref)
    # kriging

