'''
Created on Oct 5, 2022

@author: simon
'''
import numpy as np
from analysis import read_referenced_motion, save_geotiff, read_K, add_atmospheric_K
# improve phase ref
# export displacement
# test masking
year = 2022
fnunw = f'/home/simon/Work/gie/processed/Dalton_131_363/{year}/unwrapped.geo.tif'
fnK = f'/home/simon/Work/gie/processed/Dalton_131_363/{year}/K_vec.geo.tif'
fns_unw_offset = {
    2022: [],
    2019: [(7, '/home/simon/Work/gie/processed/Dalton_131_363/2019_unw_offset.gpkg')]}[year]
# fns_unw_offset = []
wavelength = 0.055
xy_ref = np.array([-148.8063, 69.1616])[:, np.newaxis]

s_obs, geospatial = read_referenced_motion(
    fnunw, xy=xy_ref, wavelength=wavelength, fns_unw_offset=fns_unw_offset)

s_obs = s_obs[1:, ...] - s_obs[0, ...][np.newaxis, ...]

ind1, ind2 = 1, -1
K, geospatial_K = read_K(fnK)
K = add_atmospheric_K(K, 0.0, wavelength=wavelength)
K_last = K[ind1, ind1, ...] + K[ind2, ind2, ...] - 2 * K[ind1, ind2, ...]
# K_last_crop, _ = geospatial.warp(K_last, geospatial_K)
save_geotiff(s_obs, geospatial, '/home/simon/Work/gie/processed/Dalton_131_363/s_obs.tif')
save_geotiff(np.sqrt(K_last)[np.newaxis, ...], geospatial, '/home/simon/Work/gie/processed/Dalton_131_363/s_std.tif')

