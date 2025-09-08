'''
Created on Sep 5, 2025

@author: simon
'''
from pathlib import Path

from analysis import (
    StefanPredictor, PredictionEnsemble, MulticlassPredictionEnsemble, read_K,
    RationalQuadraticSepDiagCovMV, add_nugget, read_geotiff_geospatial, assemble_tril,
    spatial_referencing, length_conversion, InversionProcessorIS, InversionResultsISMmap,
    InversionResultsIS, InversionProcessorGM, InversionResultsGM, InversionResultsGMMmap,
    MulticlassInversionResultsIS, MulticlassInversionResultsISMmap, MulticlassInversionResultsGM,
    MulticlassInversionResultsGMMmap, export_defo_history_hdf5, read_meta_from_json)
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple)
from forcing import load_forcing_merra_subset, parse_dates
from scripts.pathnames import paths

year = 2019
p0 = paths['processed'] / 'Pituffik/Sentinel1/' / str(year) / 'singleensemble'
import matplotlib.pyplot as plt
import numpy as np
yf_mean = np.load(p0 / 'yf_mean.npy')
e_mean = np.load(p0 / 'e_mean.npy')
plt.imshow(yf_mean[..., -1])
plt.imshow(e_mean[..., int(0.5 / 0.002)])
plt.show()