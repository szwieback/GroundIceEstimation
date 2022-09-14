'''
Created on Aug 9, 2021

@author: simon
'''

import os
import numpy as np
import pickle
import zlib
import rasterio

class Geospatial():
    def __init__(self, transform, crs, shape=None):
        self.transform = transform
        self.crs = crs
        self.shape = shape

    @classmethod
    def from_file(cls, fn):
        src = rasterio.open(fn)
        shape = (src.height, src.width)
        gsp = Geospatial(transform=src.transform, crs=src.crs, shape=shape)
        del src
        return gsp

    def rowcol(self, xy):
        # xy: lonlat for WGS84
        r, c = rasterio.transform.rowcol(self.transform, xy[0, :], xy[1, :])
        return np.stack((r, c), axis=0)

    def __eq__(self, obj):
        if not isinstance(obj, Geospatial):
            return False
        else:
            eq_transform = (self.transform == obj.transform)
            eq_shape = (self.shape == obj.shape)
            eq_crs = (self.crs == obj.crs)
            return eq_transform and eq_shape and eq_crs

def read_geotiff(fntif):
    src = rasterio.open(fntif)
    arr = src.read()
    del src
    return arr

def assemble_tril(G_vec):
    P = int(-0.5 + np.sqrt(0.25 + 2 * G_vec.shape[-1]))
    ind = np.tril_indices(P)
    G = np.zeros(tuple(G_vec.shape[:-1]) + (P, P), dtype=G_vec.dtype)
    G[(slice(None),) * (len(G_vec.shape) - 1) + ind] = G_vec
    G[(slice(None),) * (len(G_vec.shape) - 1) + (np.arange(P, dtype=np.int64),)*2] = 0
    G[(slice(None),) * (len(G_vec.shape) - 1) + (ind[1], ind[0])] += G_vec.conj()
    return G

def read_referenced_motion(fnunw, xy=None, wavelength=0.055, flip_sign=True):
    unw = read_geotiff(fnunw)
    unw *= wavelength / (4 * np.pi)
    if xy.shape[1] > 1:
        raise NotImplementedError('Only one reference point')
    geospatial = Geospatial.from_file(fnunw)
    rc = geospatial.rowcol(xy)
    unw_ref = unw[:, rc[0, 0], rc[1, 0]]
    unw -= unw_ref[:, np.newaxis, np.newaxis]
    if flip_sign: unw *= -1
    return unw, geospatial

def read_K(fntif):
    K_vec = read_geotiff(fntif)
    geospatial = Geospatial.from_file(fntif)
    K = np.moveaxis(assemble_tril(np.moveaxis(K_vec, 0, -1)), (0, 1), (-2, -1))
    return K, geospatial

def enforce_directory(path):
    path0 = os.path.dirname(path)
    if not os.path.exists(path0):
        try:
            os.makedirs(path0)
        except:
            pass
        
def save_object(obj, filename):
    enforce_directory(os.path.dirname(filename))
    if os.path.splitext(filename)[1].strip() == '.npy':
        np.save(filename, obj)
    else:
        with open(filename, 'wb') as f:
            f.write(zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))

def load_object(filename):
    if os.path.splitext(filename)[1].strip() == '.npy':
        return np.load(filename)
    with open(filename, 'rb') as f:
        obj = pickle.loads(zlib.decompress(f.read()))
    return obj

