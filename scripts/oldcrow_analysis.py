'''
Created on Feb 18, 2025

@author: simon
'''

from pathlib import Path
import numpy as np

from scripts.plot_profile import read_results, read_InSAR
from analysis import InversionResultsISMmap, save_geotiff, read_geotiff_geospatial
from scripts.oldcrowA import wavelength, var_atmo, xy_ref

pathres = Path('/home/simon/Work/gie/processed/oldcrowA/2023/')
pathstack = Path('/home/simon/Work/gie/stacks/OldCrowA/2023/geocoded/')
# ll, ur = (-166.4970, 65.3100), (-166.4100, 65.3460)
ll, ur = (-139.8650, 67.5600), (-139.799, 67.5900)
ll, ur = (-139.8650, 67.5600), (-139.799, 67.5950)

ind_K, var_thresh = -1, 8.8e-5
ind_C, C_thresh = -1, 2500


res = read_results(pathres, InversionResultsISMmap, overwrite=True)
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

# s_obs_ind, _ = geospatial_out.warp(s_obs, geospatial_K)
# s_obs_ind[:, np.isnan(e_index)[0, ...]] = np.nan
# save_geotiff(s_obs_ind, geospatial_out, fnout=pathres / 'geocoded' / 's_obs.tif')

# s_vert = s_obs_ind / np.cos(geom['ia'])
# s_vert_final_cm = s_vert[[-1], ...] * 100
#
# save_geotiff(s_obs_ind, geospatial_out, fnout=pathres / 'geocoded' / 's_obs.tif')
#
# save_geotiff(s_vert_final_cm, geospatial_out, fnout=pathres / 'geocoded' / 'subsidence_cm.tif')