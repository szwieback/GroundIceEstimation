'''
Created on Aug 9, 2021

@author: simon
'''

from pathlib  import Path
import numpy as np
import pickle
import zlib
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from collections.abc import Iterable   

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

    def rowcol(self, xy, crs=None):
        import geopandas as gpd
        # xy: lonlat for WGS84
        # an use different crs
        if isinstance(xy, np.ndarray) and len(xy.shape) >= 2:
            if crs is None:
                r, c = rasterio.transform.rowcol(self.transform, xy[0,:], xy[1,:])
                return np.stack((r, c), axis=0)
            else:
                from shapely.geometry import Point
                pts = [Point(x, y) for x, y in xy.T]
                gdf = gpd.GeoDataFrame(geometry=pts, crs=crs).to_crs(self.crs)
                return self.rowcol(gdf)
        elif isinstance(xy, gpd.GeoDataFrame):
            _xy = np.array([(x.x, x.y) for x in xy['geometry']]).T
            return self.rowcol(_xy)
        elif isinstance(xy, Iterable) or (isinstance(xy, np.ndarray) and len(xy.shape) == 1):
            return self.rowcol(np.array(xy)[:, np.newaxis], crs=crs)[:, 0]
        else:
            raise ValueError(f"xy data type not recognized")

    def xy(self, rc):
        r, c = rc[0,:], rc[1,:]
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

    def __repr__(self):
        strlist = (
            f"Transform: {repr(self.transform)}", f"CRS: {repr(self.crs)}", f"Shape: {self.shape}")
        return '\n'.join(strlist)

    def _rc_bbox(self, ll, ur):
        rc_ll = self.rowcol(np.array(ll)[:, np.newaxis])[:, 0]
        rc_ur = self.rowcol(np.array(ur)[:, np.newaxis])[:, 0]
        r = (min(rc_ll[0], rc_ur[0]), max(rc_ll[0], rc_ur[0]))
        c = (min(rc_ll[1], rc_ur[1]), max(rc_ll[1], rc_ur[1]))
        return r, c

    def cropped(self, ll, ur):
        r, c = self._rc_bbox(ll, ur)
        window = rasterio.windows.Window(c[0], r[0], c[1] - c[0], r[1] - r[0])
        transform = rasterio.windows.transform(window, self.transform)
        shape = (r[1] - r[0], c[1] - c[0])
        geospatial_out = Geospatial(transform=transform, crs=self.crs, shape=shape)
        return geospatial_out
        
    def crop(self, arr, ll=None, ur=None):
        if ll is None and ur is None:
            return arr, self
        r, c = self._rc_bbox(ll, ur)
        geospatial_out = self.cropped(ll, ur)
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
        rmethods = {'bilinear': Resampling.bilinear, 'nearest': Resampling.nearest, 'mode': Resampling.mode}
        r = rmethods[method]
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
            [Point(xy[0,:]), Point(xy[1,:])], crs=self.crs)
        s_4326 = s.to_crs(epsg='4326')
        ls = LineString([s_4326[0], s_4326[1]])
        return g.geometry_length(ls)

    @property
    def extent(self):
        xys = [self.xy(np.array([[self.shape[0], 0], self.shape]).T).T,
               self.xy(np.array([[0, self.shape[1]], self.shape]).T).T]
        return [self.distance(xy) for xy in xys]

    def rasterize(self, gdf, field='id'):
        from rasterio import features
        geom = [(shps, vals) for shps, vals in zip(gdf.geometry, gdf[field])]
        rasterized = features.rasterize(
            geom, out_shape=self.shape, fill=-1, out=None,
            transform=self.transform, default_value=-1, dtype=np.int64)[np.newaxis, ...]
        return rasterized

    def save_geotiff(self, arr, fnout, nodata=None):
        save_geotiff(arr, self, fnout, nodata=nodata)

    def _hdf5_attributes(self, nodata=np.nan, dtype='float32'):
        return assemble_hdf5_attrs(
            self.crs, self.transform, self.shape[1], 
            self.shape[0], nodata=nodata, dtype=dtype)        

def read_geotiff(fntif):
    src = rasterio.open(fntif)
    arr = src.read()
    del src
    return arr

def read_geotiff_geospatial(fntif):
    arr = read_geotiff(fntif)
    geospatial = Geospatial.from_file(fntif)
    return arr, geospatial

def save_geotiff(arr, geospatial, fnout, nodata=None, dtypename='float32'):
    meta = {
        'driver': 'GTiff', 'dtype': dtypename, 'nodata': nodata,
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

def read_referenced_InSAR(fnunw, fnK, xy_ref, wavelength=0.055, fndist=None, overwrite=False):
    # hardcodes model etc., plan to generalize (using optional kwargs) 
    from scripts.kivalina_calibration import caldict
    from analysis.interferometry import (
        add_nugget, RationalQuadraticSepDiagCovMV, spatial_referencing, length_conversion)
    unw, geospatial_unw = read_geotiff_geospatial(fnunw)
    K, geospatial_K = read_K(fnK)
    P = K.shape[0] + 1
    var_atmo = np.ones(P) * (caldict['var_rad'])  # in rad
    covmodel = RationalQuadraticSepDiagCovMV(caldict['l'], var_atmo, alpha=caldict['alpha'])
    K = add_nugget(K, caldict['nugget_speckle'])
    unw_cor, K_cor = spatial_referencing(
        unw, K, covmodel, xy_ref, geospatial_K, fndist=fndist, convert_to_length=False, 
        overwrite=overwrite)
    s_obs, K_s = length_conversion(unw_cor, K_cor, wavelength=wavelength, flip_sign=True)
    K = np.moveaxis(assemble_tril(np.moveaxis(K_s, 0, -1)), (0, 1), (-2, -1))
    assert geospatial_K == geospatial_unw
    return {'s_obs': s_obs, 'K': K}, geospatial_K

def read_referenced_motion(
        fnunw, xy=None, wavelength=0.055, flip_sign=True, fns_unw_offset=()):
    unw = read_geotiff(fnunw)
    if xy.shape[1] > 1:
        raise NotImplementedError('Only one reference point')
    from analysis.interferometry import phase_to_length
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
    unw_ref = unw[:, rc[0, 0], rc[1, 0]]
    unw -= unw_ref[:, np.newaxis, np.newaxis]
    m = phase_to_length(unw, wavelength=wavelength, flip_sign=flip_sign)
    return m, geospatial


def read_motion(
        fnunw, xy=None, wavelength=0.055, flip_sign=True, fns_unw_offset=()):
    unw = read_geotiff(fnunw)
    # if xy.shape[1] > 1:
    #     raise NotImplementedError('Only one reference point')
    def unw_to_motion(unw, wavelength=0.055, flip_sign=True):
        unw *= wavelength / (4 * np.pi)
        if flip_sign: unw *= -1
        return unw
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
    if xy is not None:
        rc = geospatial.rowcol(xy)
        unw_ref = unw[:, rc[0, 0], rc[1, 0]]
        unw -= unw_ref[:, np.newaxis, np.newaxis]
    m = unw_to_motion(unw, wavelength=wavelength, flip_sign=flip_sign)
    return m, geospatial
def K_from_K_vec(K_vec):
    return np.moveaxis(assemble_tril(np.moveaxis(K_vec, 0, -1)), (0, 1), (-2, -1))

def read_K(fntif):
    K_vec, geospatial = read_geotiff_geospatial(fntif)
    K = K_from_K_vec(K_vec)
    return K, geospatial

def enforce_directory(path):
    path.parent.mkdir(parents=True, exist_ok=True)

def save_object(obj, filename):
    pfn = Path(filename)
    enforce_directory(pfn)
    if pfn.suffix == '.npy':
        np.save(pfn, obj)
    else:
        with open(pfn, 'wb') as f:
            f.write(zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))

def load_object(filename):
    pfn = Path(filename)
    if pfn.suffix == '.npy':
        return np.load(filename)
    with open(filename, 'rb') as f:
        obj = pickle.loads(zlib.decompress(f.read()))
    return obj


def assemble_hdf5_attrs(crs, transform, width, length, band_description=None, nodata=np.nan, dtype='float32'):
    if not isinstance(transform, Affine):
        raise ValueError('transform must be a rasterio.transform.Affine')

    attrs = {
        'WIDTH': int(width),
        'LENGTH': int(length),
        'X_FIRST': float(transform.c),
        'Y_FIRST': float(transform.f),
        'X_STEP': float(transform.a),
        'Y_STEP': float(transform.e),
        'DATA_TYPE': dtype,
        'NODATA': float(nodata) if np.isfinite(nodata) else np.nan,
    }

    if isinstance(crs, CRS) and crs:
        epsg = crs.to_epsg()
        if epsg is not None:
            attrs['EPSG'] = int(epsg)
            if 32601 <= epsg <= 32660:
                attrs['UTM_ZONE'] = f'{epsg - 32600}N'
            elif 32701 <= epsg <= 32760:
                attrs['UTM_ZONE'] = f'{epsg - 32700}S'
        attrs['CRS_WKT'] = crs.to_wkt()
        attrs['UNIT'] = 'degrees' if crs.is_geographic else 'meters'

    if band_description is not None:
        attrs['BAND_DESCRIPTIONS'] = band_description
    return attrs


def hdf5_attrs_from_tif(fin_tif):
    with rasterio.open(fin_tif) as src:
        h, w = src.height, src.width
        transform = src.transform
        crs = src.crs if src.crs else None
        nodata = src.nodata if src.nodata is not None else np.nan
        dtype = str(np.dtype(src.dtypes[0]).name)
        band_desc = [x if x is not None else '' for x in src.descriptions] if any(src.descriptions) else None
        return assemble_hdf5_attrs(
            crs, transform, w, h, band_description=band_desc, nodata=nodata, dtype=dtype)


def save_hdf5(data, attrs, data_name, fnout, layer_name=None, layer_info=None):
    arr = np.asarray(data, dtype=np.dtype(attrs.get('DATA_TYPE', 'float32')))
    import h5py
    with h5py.File(fnout, 'w') as f:
        if arr.ndim == 2:
            h, w = arr.shape
            f.create_dataset(
                data_name,
                data=arr,
                dtype=arr.dtype,
                chunks=(min(512, h), min(512, w)),
                compression='lzf'
            )
        elif arr.ndim == 3:
            b, h, w = arr.shape
            f.create_dataset(
                data_name,
                data=arr,
                dtype=arr.dtype,
                chunks=(1, min(512, h), min(512, w)),
                compression='lzf'
            )        
        else:
            raise ValueError('data must be 2D (H,W) or 3D (B,H,W)')

        for k, v in attrs.items():
            if k != 'BAND_DESCRIPTIONS':
                f.attrs[k] = v
        if layer_info is not None:
            dt = h5py.string_dtype()
            ds = f.create_dataset(layer_name, (len(layer_info),), dtype=dt)
            ds[:] = np.array(layer_info, dtype=object)
        # if 'BAND_DESCRIPTIONS' in attrs:
        #     labels = list(attrs['BAND_DESCRIPTIONS'])
        #     # dt = h5py.string_dtype(encoding='utf-8')
        #     dt = h5py.string_dtype()
        #     ds = f.create_dataset('dates', (len(labels),), dtype=dt)
        #     ds[:] = np.array(labels, dtype=object)

def geotiff_to_hdf5(fin_tif, fnout_h5, data_name='data'):
    attrs = hdf5_attrs_from_tif(fin_tif)
    with rasterio.open(fin_tif) as src:
        b, h, w = src.count, attrs['LENGTH'], attrs['WIDTH']
        dtype = np.dtype(attrs['DATA_TYPE'])
        if b == 1:
            data = src.read(1, out_dtype=dtype)
        else:
            data = np.empty((b, h, w), dtype=dtype)
            for i in range(1, b + 1):
                data[i - 1] = src.read(i, out_dtype=dtype)
    save_hdf5(data, attrs, data_name, fnout_h5)

def hdf5_to_geotiff(fin_h5, fout_tif, dataset='yf_mean', layer_name='dates'):
    import h5py
    with h5py.File(fin_h5, 'r') as f:
        # if dataset is None:
        #     keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and k.lower() != 'dates']
        #     if not keys:
        #         raise ValueError('No data dataset found in HDF5.')
        #     dataset = keys[0]
        arr = f[dataset][()]
        arr = np.moveaxis(arr, 2, 0)
        attrs = f.attrs

        if arr.ndim == 2:
            count, height, width = 1, arr.shape[0], arr.shape[1]
        elif arr.ndim == 3:
            count, height, width = arr.shape[0], arr.shape[1], arr.shape[2]
        else:
            raise ValueError(f'Unsupported array shape: {arr.shape}')

        transform = Affine(float(attrs['X_STEP']), 0.0, float(attrs['X_FIRST']),
                           0.0, float(attrs['Y_STEP']), float(attrs['Y_FIRST']))

        crs = None
        epsg = attrs.get('EPSG', None)
        if epsg is not None:
            try: crs = CRS.from_epsg(int(epsg))
            except Exception: crs = None
        if crs is None:
            wkt = attrs.get('CRS_WKT', None) or attrs.get('crs_wkt', None)
            if wkt:
                try: crs = CRS.from_wkt(wkt)
                except Exception: crs = None

        nod = attrs.get('NoDataValue', np.nan)
        nodata = None if (isinstance(nod, float) and not np.isfinite(nod)) else float(nod)

        dtype = str(arr.dtype)
        profile = {
            'driver': 'GTiff',
            'height': int(height),
            'width': int(width),
            'count': int(count),
            'dtype': dtype,
            'transform': transform,
            'crs': crs,
            'compress': 'LZW',
            'tiled': True
        }
        if nodata is not None:
            profile['nodata'] = nodata

        with rasterio.open(fout_tif, 'w', **profile) as dst:
            if count == 1 and arr.ndim == 2:
                dst.write(arr, 1)
            else:
                dst.write(arr)

            labels = None
            if layer_name in f and isinstance(f[layer_name], h5py.Dataset):
                dset = f[layer_name][()]
                labels = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in dset]
            elif 'BAND_DESCRIPTIONS' in attrs:
                bd = attrs['BAND_DESCRIPTIONS']
                if isinstance(bd, (list, tuple, np.ndarray)):
                    labels = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in bd]

            if labels and len(labels) == count:
                for i, lab in enumerate(labels, start=1):
                    dst.set_band_description(i, lab)

def export_defo_history_hdf5(
        s_obs, pout, geospatial, geom, dates_obs_str=None, K=None, flip_sign=True, 
        fn_defo='defo_history.h5', fn_K_diag='defo_history_covariance_diagonal.h5'):
    attributes = geospatial._hdf5_attributes()
    attributes['INC_ANGLE'] = geom['ia'] * 180 / np.pi #degrees
    if flip_sign:
        s_obs = s_obs * (-1) # so subsidence is negative
    save_hdf5(s_obs, attributes, 'data', pout / fn_defo, layer_name='dates', layer_info=dates_obs_str)
    if K is not None:
        K_diag = np.moveaxis(np.diagonal(K), -1, 0) # diagonal insanely messes up the ordering
        save_hdf5(
            K_diag, attributes, 'variance', pout / fn_K_diag, layer_name='dates', layer_info=dates_obs_str)
        
def get_dates_obs_str(dailytemp, ind_scenes):
    from datetime import timedelta
    dates_obs = [
        (dailytemp.index[0] + timedelta(days=ind_scene)).strftime('%Y%m%d') for ind_scene in ind_scenes]    
    
def read_meta_from_json(fnmeta):
    import json
    with open(fnmeta, 'r') as file:
        meta = json.load(file)
        meta['geom'] = {'ia': float(meta.pop('ia')) * np.pi / 180} # to radians
        meta['wavelength'] = float(meta['wavelength'])
    return meta