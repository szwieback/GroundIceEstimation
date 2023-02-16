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
    
    @classmethod
    def plate_carree(cls, bbox, dlat=-2e-4, dlon=None):
        from rasterio.transform import Affine
        from rasterio.crs import CRS
        crs = CRS.from_epsg(4326)
        assert dlat < 0
        if dlon is None: dlon = np.abs(dlat) / np.cos(bbox[0] * np.pi / 180)
        transform = Affine(dlon, 0.0, bbox[2], 0.0, dlat, bbox[1])
        shape = (
            int((bbox[1] - bbox[0]) / np.abs(dlat)),
            int((bbox[3] - bbox[2]) / np.abs(dlon)))
        gsp = Geospatial(transform=transform, crs=crs, shape=shape)
        return gsp
    
    @property
    def rowcol_grids(self):
        return (np.arange(self.shape[0]), np.arange(self.shape[1]))

    def rowcol(self, xy):
        # xy: lonlat for WGS84
        r, c = rasterio.transform.rowcol(self.transform, xy[0,:], xy[1,:])
        return np.stack((r, c), axis=0)
    
    def xy(self, rc):
        r, c = rc[0, :], rc[1, :]
        return np.array(rasterio.transform.AffineTransformer(self.transform).xy(r, c))

    @property
    def xy_raster(self):
        from affine import Affine
        assert isinstance(self.transform, Affine)
        sa, sb, sc, sd, se, sf, _, _, _ = self.transform
        # diverging indexing convention (column, row)
        c, r = np.meshgrid(range(self.shape[0]), range(self.shape[1]), indexing='ij')
        # center of pixel convention
        c, r = c.astype(np.float64) + 0.5, r.astype(np.float64) + 0.5
        x = r * sa + c * sb + sc
        y = r * sd + c * se + sf
        xy = np.stack((x, y), axis=-1)
        return xy

    def __eq__(self, obj):
        if not isinstance(obj, Geospatial):
            return False
        else:
            eq_transform = (self.transform == obj.transform)
            eq_shape = (self.shape == obj.shape)
            eq_crs = (self.crs == obj.crs)
            return eq_transform and eq_shape and eq_crs

    def __str__(self):
        strlist = (
            f"Transform: {self.transform}", f"CRS: {self.crs}", f"Shape: {self.shape}")
        return '\n'.join(strlist)

    def _rc_bbox(self, ll, ur):
        rc_ll = self.rowcol(np.array(ll)[:, np.newaxis])[:, 0]
        rc_ur = self.rowcol(np.array(ur)[:, np.newaxis])[:, 0]
        r = (min(rc_ll[0], rc_ur[0]), max(rc_ll[0], rc_ur[0]))
        c = (min(rc_ll[1], rc_ur[1]), max(rc_ll[1], rc_ur[1]))
        return r, c

    def crop(self, arr, ll=None, ur=None):
        if ll is None and ur is None:
            return arr, self
        r, c = self._rc_bbox(ll, ur)
        window = rasterio.windows.Window(c[0], r[0], c[1] - c[0], r[1] - r[0])

        transform = rasterio.windows.transform(window, self.transform)

        shape = (r[1] - r[0], c[1] - c[0])
        geospatial_out = Geospatial(transform=transform, crs=self.crs, shape=shape)
        arr_out = arr[..., r[0]:r[1], c[0]:c[1]]
        return arr_out, geospatial_out

    def upscaled(self, upscale=None):
        if upscale is None: return self
        _us = int(upscale)
        assert _us > 0
        shape = tuple((np.array(self.shape) * _us).astype(np.uint64))
        a, b, c, d, e, f, g, h, i = self.transform
        transform = rasterio.Affine(a / _us, b / _us, c, d / _us, e / _us, f)
        return Geospatial(transform, self.crs, shape)

    def warp(
            self, arr_in, geospatial_in, method='bilinear', dtype=np.float32, upscale=None):
        from rasterio.warp import reproject, Resampling
        r = {'bilinear': Resampling.bilinear}[method]  # implement others
        _gs = self.upscaled(upscale=upscale)
        arr_out = np.zeros(arr_in.shape[:-2] + _gs.shape, dtype=dtype)
        reproject(
            arr_in, arr_out, src_transform=geospatial_in.transform,
            src_crs=geospatial_in.crs, dst_transform=_gs.transform, dst_crs=_gs.crs,
            resampling=r)
        return arr_out, _gs

    def warp_from_file(self, fn, method='bilinear', dtype=np.float32, upscale=None):
        arr, geospatial = read_geotiff_geospatial(fn)
        return self.warp(arr, geospatial, method=method, dtype=dtype, upscale=upscale)

    def distance(self, xy):
        import geopandas as gpd
        from shapely.geometry import Point, LineString
        from pyproj import Geod
        g = Geod(ellps='WGS84')
        s = gpd.GeoSeries(
            [Point(xy[0, :]), Point(xy[1, :])], crs=self.crs)
        s_4326 = s.to_crs(epsg='4326')
        ls = LineString([s_4326[0], s_4326[1]])
        return g.geometry_length(ls)

    @property
    def extent(self):
        xys = [self.xy(np.array([[self.shape[0], 0], self.shape]).T).T,
               self.xy(np.array([[0, self.shape[1]], self.shape]).T).T]
        return [self.distance(xy) for xy in xys]
        
        

def read_geotiff(fntif):
    src = rasterio.open(fntif)
    arr = src.read()
    del src
    return arr

def read_geotiff_geospatial(fntif):
    arr = read_geotiff(fntif)
    geospatial = Geospatial.from_file(fntif)
    return arr, geospatial

def save_geotiff(arr, geospatial, fnout, nodata=None):
    meta = {
        'driver': 'GTiff', 'dtype': 'float32', 'nodata': nodata,
        'width': geospatial.shape[1], 'height': geospatial.shape[0],
        'count': arr.shape[0], 'crs': geospatial.crs, 'transform': geospatial.transform}
    enforce_directory(fnout)
    with rasterio.open(fnout, 'w', **meta) as dst:
        dst.write(arr)

def assemble_tril(G_vec):
    P = int(-0.5 + np.sqrt(0.25 + 2 * G_vec.shape[-1]))
    ind = np.tril_indices(P)
    G = np.zeros(tuple(G_vec.shape[:-1]) + (P, P), dtype=G_vec.dtype)
    G[(slice(None),) * (len(G_vec.shape) - 1) + ind] = G_vec
    G[(slice(None),) * (len(G_vec.shape) - 1) + (np.arange(P, dtype=np.int64),) * 2] = 0
    G[(slice(None),) * (len(G_vec.shape) - 1) + (ind[1], ind[0])] += G_vec.conj()
    return G

def vectorize_tril(G):
    # ..., P, P to ..., P * (P + 1) / 2
    P = G.shape[-1]
    assert G.shape[-2] == P
    ind = np.tril_indices(P)
    ind_ = (slice(None),) * (len(G.shape) - 2) + ind
    G_vec = G[ind_]
    return G_vec

def read_referenced_motion(
        fnunw, xy=None, wavelength=0.055, flip_sign=True, fns_unw_offset=()):
    unw = read_geotiff(fnunw)
    if xy.shape[1] > 1:
        raise NotImplementedError('Only one reference point')
    geospatial = Geospatial.from_file(fnunw)
    if len(fns_unw_offset) >= 1:
        import geopandas as gpd
        from rasterio import features
        for scene, fn in fns_unw_offset:
            offset = gpd.read_file(fn).to_crs(geospatial.crs)
            geom = [(shps, offs) for shps, offs in zip(offset.geometry, offset['offset'])]
            rasterized = features.rasterize(
                geom, out_shape=geospatial.shape, fill=0, out=None, 
                transform=geospatial.transform, default_value=0, dtype=np.int64)
            unw[scene:, ...] += rasterized * 2 * np.pi
    rc = geospatial.rowcol(xy)
    unw *= wavelength / (4 * np.pi)
    unw_ref = unw[:, rc[0, 0], rc[1, 0]]
    unw -= unw_ref[:, np.newaxis, np.newaxis]
    if flip_sign: unw *= -1
    return unw, geospatial

def read_K(fntif):
    K_vec, geospatial = read_geotiff_geospatial(fntif)
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

if __name__ == '__main__':
    pathres = '/home/simon/Work/gie/processed/kivalina/2019/hadamard/inversion/'
    geospatial = load_object(os.path.join(pathres, 'geospatial.p'))
    
    print(geospatial.extent)
    