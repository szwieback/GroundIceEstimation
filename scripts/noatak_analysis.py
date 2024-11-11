'''
Created on Nov 5, 2024

@author: simon
'''
from pathlib import Path
import numpy as np

from scripts.plot_profile import read_results, read_InSAR
from analysis import InversionResultsISMmap, save_geotiff, read_geotiff_geospatial
from scripts.noatak import wavelength, xy_ref, var_atmo

pathres = Path('/home/simon/Work/gie/processed/noatak/2019/hadamard/')
pathstack = Path('/home/simon/Work/gie/stacks/Noatak/2019/hadamard/geocoded/')
ll, ur = (-163.097, 67.522), (-162.952, 67.590)
ind_K, var_thresh = -1, (0.0068)**2
ind_C, C_thresh = 1, 2000


res = read_results(pathres, InversionResultsISMmap, overwrite=False)
s_obs, K, geospatial_K = read_InSAR(pathstack, wavelength, var_atmo=var_atmo, xy_ref=xy_ref)
K_ind = K[ind_K, ind_K, ...]

geospatial_out = res['geospatial'].cropped(ll, ur)

C_diag, geospatial_C = read_geotiff_geospatial(pathstack / 'C_diag.geo.tif')

K_ind, _ = geospatial_out.warp(K[ind_K, ind_K, ...][np.newaxis, ...], geospatial_K)
C_ind, _ = geospatial_out.warp(C_diag[ind_C, ...][np.newaxis, ...], geospatial_C)
e_index, _ = geospatial_out.warp(res['e_mean_period_mean'][np.newaxis, ..., 0], res['geospatial'])
empq = np.moveaxis(res['e_mean_period_quantile'][..., 0, :], -1, 0)
e_index_q, _ = geospatial_out.warp(empq, res['geospatial'])
e_index[np.logical_or(np.isnan(K_ind), K_ind > var_thresh)] = np.nan
e_index[C_ind < C_thresh] = np.nan
e_index_q[:, np.isnan(e_index)[0, ...]] = np.nan

save_geotiff(e_index, geospatial_out, fnout=pathres / 'geocoded' / 'e_mean_period.tif')
save_geotiff(e_index_q, geospatial_out, fnout=pathres / 'geocoded' / 'e_mean_period_quantile.tif')


# save_geotiff(e_index, geospatial_K, fnout=pathres / 'geocoded' / 'e_mean_period.tif')

# save_geotiff(res['e_mean_period_mean'][np.newaxis, ..., 0], geospatial=res['geospatial'], fnout=pathres / 'geocoded' / 'e_mean_period.tif')

# to do:
# mask low coherence / nan