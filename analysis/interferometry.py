'''
Created on Sep 14, 2022

@author: simon
'''

import numpy as np
from scipy.linalg import block_diag
import warnings
import os
from abc import ABC, abstractmethod

from analysis import save_object, load_object, save_geotiff

wvl0 = 0.055

def phase_to_length_factor(wavelength=wvl0):
    # two-way prop
    conv = wavelength / (4 * np.pi)
    return conv

def phase_to_length(phase, wavelength=wvl0, variance=False):
    conv = phase_to_length_factor(wavelength)
    power = 2 if variance else 1
    return phase * (conv ** power)

def var_atmo_conversion(var_atmo, to_length=True, wavelength=wvl0, var_atmo_phase=False):
    warnings.warn("var_atmo_conversion deprecated")
    conv = phase_to_length_factor(wavelength)
    _var_atmo = var_atmo
    if var_atmo_phase and to_length:
        _var_atmo = var_atmo * (conv ** 2)
    elif not to_length and not var_atmo_phase:
        _var_atmo = var_atmo / (conv ** 2)
    return _var_atmo

def add_atmospheric_K(K, var_atmo, wavelength=wvl0, to_length=True, var_atmo_phase=False):
    warnings.warn("add_atmospheric_K deprecated")
    conv = phase_to_length_factor(wavelength)
    if to_length:
        K *= conv ** 2
    K_eye = np.eye(K.shape[0])[(Ellipsis,) + (None,) * (len(K.shape) - 2)]
    _var_atmo = var_atmo_conversion(
        var_atmo, to_length=to_length, wavelength=wavelength, var_atmo_phase=var_atmo_phase)
    K += _var_atmo * (np.ones_like(K) + K_eye)
    return K

def combined_K(K_speckle, covmodel):
    # var_atmo assumed stationary
    P = K_speckle.shape[0] + 1
    A = _phase_history_matrix(P)
    C_atmo = covmodel.covariance_multivariate
    K_atmo = np.einsum('ij,jk...,lk ->il...', A, C_atmo, A)
    K_comb = K_speckle + K_atmo[(Ellipsis,) + (np.newaxis,) * (len(K_speckle.shape) - 2)]
    return K_comb

def compute_distances(xy_point, xy_ref, geospatial):
    def _compute_distance_jref(jref):
        _xy_ref = xy_ref[:, jref][:, np.newaxis]
        xyl = np.concatenate((xy_point, _xy_ref), axis=1)
        return geospatial.distance(xyl.T)
    return np.array([_compute_distance_jref(jref) for jref in range(xy_ref.shape[1])])

def distance_to_ref(
        geospatial, xy_ref, fndist=None, geospatial_ref=None, overwrite=False, njobs=-2, block_size=256):
    # uses projected c.s.; slow, but probably not a bottleneck
    import geopandas as gpd
    from shapely.geometry import Point
    gsp_ref = geospatial_ref if geospatial_ref is not None else geospatial
    s_ref = gpd.GeoSeries([Point(xy_ref[:, jref]) for jref in range(xy_ref.shape[1])], crs=gsp_ref.crs)

    crs_dist = s_ref.estimate_utm_crs()
    s_ref_proj = s_ref.to_crs(crs_dist)
    x_ref_proj, y_ref_proj = np.array(s_ref_proj.x), np.array(s_ref_proj.y)
    if fndist is None or overwrite or not os.path.exists(fndist):
        print('reprocessing')
        xy_raster = geospatial.xy_raster
        if njobs not in [0, 1, None]:
            from joblib import Parallel, delayed
            xy_blocks = np.array_split(np.reshape(xy_raster, (-1, 2)), block_size, axis=0)
            def _process_block(xy_block):
                s_block = gpd.GeoSeries(
                    [Point(_xy) for _xy in xy_block], crs=geospatial.crs)
                s_block_proj = s_block.to_crs(crs_dist)
                x_block, y_block = np.array(s_block_proj.x), np.array(s_block_proj.y)
                dist = np.stack(
                [np.sqrt(((x_block - x) ** 2 + (y_block - y) ** 2)) for x, y in zip(x_ref_proj, y_ref_proj)],
                axis=-1)
                return dist
            dist = np.concatenate(
                Parallel(n_jobs=njobs)(delayed(_process_block)(block) for block in xy_blocks))
        else:
            s_raster = gpd.GeoSeries(
                [Point(_xy) for _xy in np.reshape(xy_raster, (-1, 2))], crs=geospatial.crs)
            s_raster_proj = s_raster.to_crs(crs_dist)
            x_raster, y_raster = np.array(s_raster_proj.x), np.array(s_raster_proj.y)
            dist = np.stack(
                [np.sqrt((x_raster - x) ** 2 + (y_raster - y) ** 2) for x, y in zip(x_ref_proj, y_ref_proj)],
                axis=-1)
        distances = np.reshape(dist, geospatial.shape + (-1,))
        if fndist is not None:
            print(f'overwriting {fndist}')
            dict_out = {
                'geospatial': geospatial, 'xy_ref': xy_ref, 'geospatial_ref': gsp_ref,
                'distances': distances}
            save_object(dict_out, fndist)
    else:
        dict_res = load_object(fndist)
        assert geospatial == dict_res['geospatial']
        np.testing.assert_allclose(xy_ref, dict_res['xy_ref'])
        assert gsp_ref == dict_res['geospatial_ref']
        distances = dict_res['distances']
    return distances

def extract_reference(K_speckle, unw, dist, geospatial, xy_ref, covmodel):
    # to do: proper object structure for atmosphere
    # returns phase history and Kronecker-structured covariance (speckle + atmos)
    # get K_ref from xy_ref
    rc_ref = geospatial.rowcol(xy_ref)
    K_ref_stacked = np.stack([K_speckle[..., _r, _c] for _r, _c in rc_ref.T], axis=0)
    unw_ref = np.concatenate([unw[..., _r, _c] for _r, _c in rc_ref.T], axis=0)
    P = K_speckle.shape[0] + 1
    K_ref = K_reference_block(K_ref_stacked)
    # get distances
    dist_matrix = np.array([dist[_r, _c,:] for _r, _c in rc_ref.T])
    dist_matrix -= np.diag(np.diag(dist_matrix))  # set diagonals to zero (subpixel localization issue)
    C_comb = covmodel.covariance(dist_matrix)
    A_block = _phase_history_matrix(P, blocks=dist_matrix.shape[0])
    # to phase differences (w.r.t first)
    K_ref_atmo = np.einsum('ij,jk...,lk ->il...', A_block, C_comb, A_block)
    K_ref_comb = K_ref_atmo + K_ref
    return unw_ref, K_ref_comb

def extract_cross_covariance(dist, covmodel):
    # phase difference
    C_cross = covmodel.cross_covariance(dist)
    P = covmodel.N_variables
    A, A_block = _phase_history_matrix(P), _phase_history_matrix(P, blocks=dist.shape[-1])
    K_cross = np.einsum('ij,...jk,lk ->...il', A_block, C_cross, A, optimize=True)
    return K_cross

def _phase_history_matrix(P, reference=0, blocks=1):
    A = np.zeros((P - 1, P))
    if not isinstance (reference, int): raise NotImplementedError
    A[:, reference] = -1
    for p in range(P):
        if p < reference:
            A[p, p] = 1
        elif p > reference:
            A[p - 1, p] = 1
    if blocks != 1:
        # to do: sparse matrix for longer stacks
        _A = A.copy()
        A = block_diag(*((_A,) * blocks))
    return A

def K_from_phase_covariance(C_phase, reference=0):
    P = C_phase.shape[0]
    assert C_phase.shape[1] == P
    A = _phase_history_matrix(P, reference=reference)
    K = np.einsum('ij,jk...,lk ->il...', A, C_phase, A)
    # test = np.einsum('ij,jk...->...ik', A, C_phase)
    # print(test.shape)
    return K

def K_reference_block(K):
    # Kronecker product with identity matrix (assumes speckle spatially uncorrelated)
    # input: K [N, P-1, P-1] where N is number of reference points
    # output: K [PN-p, PN-P]
    return block_diag(*[_K for _K in K])

def kriging_inverses(K_ref_comb, P):
    # ordinary kriging matrices (excluding Lagrange multiplier row) for each phase diff separately
    krig_matrices = []
    for p in range(P - 1):
        _K_ref_comb = K_ref_comb[p::P - 1, p::P - 1]
        krig_matrix = np.ones((_K_ref_comb.shape[0] + 1,) * 2)
        krig_matrix[:-1,:-1] = _K_ref_comb[:,:]
        krig_matrix[-1, -1] = 0
        krig_matrix = np.linalg.pinv(krig_matrix, hermitian=True)
        krig_matrices.append((krig_matrix[:-1,:-1], krig_matrix[:-1, -1]))
    return krig_matrices

def weight_matrix(K_cross, krig_matrices):
    W = np.zeros(K_cross.shape[:-2] + (K_cross.shape[-1],) + (K_cross.shape[-2],))
    P = len(krig_matrices) + 1
    for p in range(P - 1):
        K_cross_p = K_cross[..., p::P - 1, p]
        krig_matrix, krig_offset = krig_matrices[p]
        _W = np.einsum('ij, ...j->...i', krig_matrix, K_cross_p, optimize=True)
        _W += krig_offset[(np.newaxis,) * (len(_W.shape) - 1) + (Ellipsis,)]
        assert np.max(np.abs(np.sum(_W, axis=-1) - 1)) < 1e-12
        W[..., p, p::P - 1] = _W
    return W

def _kriging_reference(unw, K_comb, K_cross, W, unw_ref, K_ref_comb, vectorize=True):
    # modifies in place
    from analysis import vectorize_tril
    unw -= np.einsum('...ij,j->...i', W, unw_ref, optimize=True)
    K_comb += np.einsum('...ij,jk,...lk->...il', W, K_ref_comb, W, optimize=True)
    dK = np.einsum('...ij,...jk->...ik', W, K_cross, optimize=True)
    K_comb -= dK + np.moveaxis(dK, -2, -1)
    if vectorize: K_comb = vectorize_tril(K_comb)
    return unw, K_comb

class CovMV(ABC):
    @abstractmethod
    def __init__(self):
        pass

class SepDiagCovMV(CovMV):
    # assumes separable structure, with multivariate cov diagonal with variances var
    @abstractmethod
    def __init__(self, var):
        self.var = var

    @property
    def covariance_multivariate(self):
        return np.diag(self.var)

    @abstractmethod
    def covariance_spatial(self, dist):
        pass

    @property
    def N_variables(self):
        return len(self.var)

    def covariance(self, dist):  # Kronecker product structure
        C_spatial = self.covariance_spatial(dist)
        C_mv = self.covariance_multivariate
        # gives T0x0, T1x0, ...T0x1, T1x1,
        C_comb = np.kron(C_spatial, C_mv)  # use kronecker for same spatial
        return C_comb

    def cross_covariance(self, dist):
        C_spatial = self.covariance_spatial(dist)
        C_mv = self.covariance_multivariate
        newaxes = (np.newaxis,) * (len(dist.shape) - 1)
        C_cross = np.kron(C_spatial[(Ellipsis,) + newaxes], C_mv[newaxes + (Ellipsis,)])
        return C_cross

class RationalQuadraticSepDiagCovMV(SepDiagCovMV):
    # stationary
    def __init__(self, l, var, alpha=2):
        self.l = l
        self.var = var
        self.alpha = alpha

    def covariance_spatial(self, dist):
        cov = (1 + (dist ** 2) / (2 * self.alpha * (l ** 2))) ** (-self.alpha)
        return cov

def spatial_referencing(
        unw, K, covmodel, xy_ref, geospatial, n_jobs=7, fnunw=None, fnK=None, fndist=None,
        convert_to_length=True, overwrite=False):
    # covmodel in length; unw, K in phase unless convert_to_length is False
    from joblib import Parallel, delayed
    P = K.shape[0] + 1
    unw -= np.nanmean(unw, axis=(1, 2))[:, np.newaxis, np.newaxis]
    if convert_to_length:
        K = phase_to_length(K, wavelength=wvl, variance=True)
        unw = phase_to_length(unw, wavelength=wvl, variance=False)        
    K_comb = combined_K(K, covmodel)
    dist = distance_to_ref(geospatial, xy_ref, fndist=fndist, overwrite=False)
    unw_ref, K_ref_comb = extract_reference(K, unw, dist, geospatial, xy_ref, covmodel)
    krig_matrices = kriging_inverses(K_ref_comb, P)
    def _phase_history_reference_row(row):
        _unw = np.moveaxis(unw[:, row,:], 0, 1).copy()
        _dist = dist[row, ...]
        _K_comb = np.moveaxis(K_comb[..., row,:], (0, 1), (-2, -1)).copy()
        _K_cross = extract_cross_covariance(_dist, covmodel)
        W = weight_matrix(_K_cross, krig_matrices)
        _unw_cor, _K_cor_vec = _kriging_reference(
            _unw, _K_comb, _K_cross, W, unw_ref, K_ref_comb)
        return _unw_cor, _K_cor_vec
    rows = range(unw.shape[-2])
    res = Parallel(n_jobs=n_jobs)(delayed(_phase_history_reference_row)(row) for row in rows)
    unw_cor = np.moveaxis(np.stack([r[0] for r in res], axis=0), -1, 0)
    K_cor = np.moveaxis(np.stack([r[1] for r in res], axis=0), -1, 0)
    if fnunw is not None:
        save_geotiff(unw_cor, geospatial, fnunw)
    if fnK is not None:
        save_geotiff(K_cor, geospatial, fnK)
    return unw_cor, K_cor


if __name__ == '__main__':
    import os
    from analysis import (read_K, read_geotiff_geospatial)
    path0 = f'/home/simon/Work/gie/processed/kivalina/2019'
    geom = {'ia': 39.29 / 180 * np.pi}
    wavelength = 0.055
    var_atmo = (4e-3) ** 2
    xy_ref = np.array([[-164.7300, 67.8586], [-164.4091, 67.7663], [-164.9063, 67.9318]]).T
    l = 1e4
    wvl = wvl0

    fnunw = os.path.join(path0, 'unwrapped.geo.tif')
    fnK = os.path.join(path0, 'K_vec.geo.tif')
    K, geospatial_K = read_K(fnK)
    unw, geospatial_unw = read_geotiff_geospatial(fnunw)
    assert geospatial_unw == geospatial_K
    
    P = K.shape[0] + 1

    var_atmo = np.ones(P) * ((0.03) ** 2)  # in m
    covmodel = RationalQuadraticSepDiagCovMV(l, var_atmo)
    
    fndist = os.path.join(path0, 'distance.p')
    unw_cor, K_cor = spatial_referencing(
        unw, K, covmodel, xy_ref, geospatial_K, fndist=fndist)
    from analysis import assemble_tril
    K = assemble_tril(K_cor[:, 600, 400])


