'''
Created on Feb 29, 2024

@author: simon
'''
from analysis import Geospatial, read_geotiff_geospatial, save_geotiff
import numpy as np
import os

def check_lut(classes):
    for clist in classes.values():
        for v in clist:
            if sum([v in cl for cl in classes.values()]) > 1:
                raise ValueError(f'Ambiguous assignment of original class {v}')

def reclassify(fnlc, eclasses, geospatial, fnout=None, overwrite=False,nodata=-1):
    if overwrite or fnout is None or not os.path.exists(fnout):
        dtype, dtypename = np.int16, 'int16'
        check_lut(eclasses)
        lc_rs, _ = geospatial.warp_from_file(fnlc, dtype=np.int16, method='mode')
        ec = np.full((1,) + geospatial.shape, nodata, dtype=dtype)
        for cn, clist in eclasses.items():
            np.putmask(ec, np.isin(lc_rs, clist), cn)
        if fnout is not None:
            save_geotiff(ec, geospatial, fnout, dtypename=dtypename)
    else:
        ec, geospatial_ec = read_geotiff_geospatial(fnout)
        if not geospatial == geospatial_ec: raise ValueError(f'Geospatial of {fnout} inconsistent')
    return ec 

if __name__ == '__main__':
    fnlc = '/home/simon/Work/gie/ancillary/TNC/ecosystems_northern_alaska_jorgenson_2010.tif'
    fnunw = '/home/simon/Work/gie/processed/Dalton_131_363/2023/unwrapped.geo.tif'
    unw, geospatial = read_geotiff_geospatial(fnunw)

    eclasses = {0: (1, 3, 11, 12, 13, 14, 15, 18, 23, 32, 41, 43, 44, 45, 46, 47, 48, 112, -99),
               1: (2, 21, 25, 26, 33, 34, 35)}

    fnlcrs = '/home/simon/Work/gie/processed/lc.tif'
    fnec = '/home/simon/Work/gie/processed/ec.tif'


    ec = reclassify(fnlc, eclasses, geospatial, fnout=fnec, overwrite=False)
    print(ec.shape)
    
    # next steps:
    # Happy Valley
    # new IP with ensembledict, separate call with ensembleclass img


