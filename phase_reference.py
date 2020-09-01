'''
Created on Dec 2, 2019

@author: simon
'''
import numpy as np
import os
from InterferometricSpeckle.probability import (ProcessedOptimizationStack,
                                                covariance_matrix_from_sample)
from InterferometricSpeckle.storage import (filename, paths, save_object, load_object)

wavelength = 0.056
model_inference = 'GaussianBandedInversePlusRankOneU2'

def ps_candidates(y, lam=0.3):
    tempmean = np.mean(np.abs(y), axis=-2)
    D = np.sqrt(np.mean((np.abs(y) - tempmean[..., np.newaxis, :]) ** 2, axis=-2)) / tempmean
    B = np.median(tempmean) / tempmean
    return (1 - lam) * D + lam * B

def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    # return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])
    return np.array(np.unravel_index(idx, arr.shape))[:, range(k)].transpose()

def referenced_cphase(phi, location, y, radius0=20, criterion0=0.1, N_candidate=0,
                      location_reference=None):
    # approximately square pixels assumed
    if location_reference is None:
        ps_criterion = ps_candidates(y)
        pos_x = np.arange(y.shape[0])[:, np.newaxis]
        pos_y = np.arange(y.shape[1])[np.newaxis, :]
        dist = np.sqrt((pos_x - location[0]) ** 2 + (pos_y - location[1]) ** 2)
        criterion_dist = ps_criterion / criterion0 + (dist / radius0)[..., np.newaxis]
        idx = get_indices_of_k_smallest(criterion_dist, N_candidate + 1)[N_candidate, :]
        print(idx)
        y_ps = y[idx[0], idx[1], :, idx[2]]
        phi_ps = (y_ps * y_ps[0].conj())[1:]
    else:
        phi_ps = phi[location_reference[0], location_reference[1], ...]
    cphi_ps = np.exp(1j * phi_ps)
    cphi_corr = np.exp(1j * phi) * cphi_ps[np.newaxis, np.newaxis, :].conj()
    return cphi_corr

def displacement_time_series(location=(62, 189), radius0=20, criterion0=0.1, N_candidate=0,
        stackname='dalton', model_inference=model_inference, method='full', paths=paths,
        location_reference=None, overwrite=False):

    fnpickle = filename(
        fields={'batch': 'preproc'}, subcategories=(stackname,), meta=None,
        category='processed', paths=paths)
    if not os.path.exists(fnpickle) or overwrite:
        fields_out = {'batch': 'processed', 'L': 100, 'phase-closure': True,
                      'cohcorr': 'ccpr'}
        fnstack = filename(fields=fields_out, meta=None, category='processed',
                         subcategories=(stackname,), paths=paths)
        pospc = ProcessedOptimizationStack.from_file(fnstack)
        phi = pospc.by_model('phases', model_name=model_inference, is_short_name=False)
        y = pospc.as_array(pospc.y)
        C_obs = covariance_matrix_from_sample(y)
        save_object({'C_obs': C_obs, 'phi': phi}, fnpickle)
    else:
        dictphi = load_object(fnpickle)
        C_obs, phi = dictphi['C_obs'], dictphi['phi']
    cphi_corr = referenced_cphase(
        phi, location, 0, radius0=radius0, criterion0=criterion0, N_candidate=N_candidate,
        location_reference=location_reference)
    print(location, location_reference, cphi_corr.shape)
    cphi_location = cphi_corr[location[0], location[1], ...]
    phi_location = np.unwrap(np.angle(cphi_location))
    disp_location = (phi_location / np.pi) * wavelength
    try:
        ind_model = [sm.name for sm in pospc.specklem_inference].index(model_inference)
        sm = pospc.specklem_inference[ind_model]
        theta = pospc.by_model('theta', model_name=model_inference, is_short_name=False)
        print(y.shape, phi.shape, C_obs.shape, theta.shape)
        theta_location = theta[location[0], location[1], ...]
        y_location = y[location[0], location[1], ...]
        C_location = sm.phase_history_covariance(
            theta_location, y=y_location, method=method, average=False)
        C_location = C_location * (wavelength / np.pi) ** 2
        C_location = C_location * 2  # due to oversampling
        import warnings
        warnings.warn('Hack to account for oversampling', DeprecationWarning)
    except:
        C_location = None
    return {'disp': disp_location, 'C': C_location}
