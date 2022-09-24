'''
Created on Sep 19, 2022

@author: simon
'''
import os
import numpy as np
from analysis.ioput import Geospatial, save_geotiff, read_geotiff, save_object, load_object
from analysis.inversion import InversionResults

class ProfileInterpolator():
    def __init__(self, geospatial, xy_start, xy_end, steps=128):
        self.geospatial = geospatial
        self.xy_start = xy_start
        self.xy_end = xy_end
        self.steps = steps

    @property
    def _rowcol_endpoints(self):
        return self.geospatial.rowcol(np.stack((self.xy_start, self.xy_end), axis=1))

    def _interpolator(self, arr):
        from scipy.interpolate import RegularGridInterpolator
        rowcol_grids = self.geospatial.rowcol_grids
        return RegularGridInterpolator(rowcol_grids, arr)

    def interpolate(self, arr):
        _ip = self._interpolator(arr)
        rc = self._rowcol_endpoints
        rc_steps = np.stack(
            [np.linspace(rc[ji, 0], rc[ji, 1], num=self.steps) for ji in range(2)], axis=1)
        return _ip(rc_steps)

    @property
    def distance(self):
        import geopandas as gpd
        from shapely.geometry import Point, LineString
        from pyproj import Geod
        g = Geod(ellps='WGS84')
        s = gpd.GeoSeries(
            [Point(self.xy_start), Point(self.xy_end)], crs=self.geospatial.crs)
        s_4326 = s.to_crs(epsg='4326')
        ls = LineString([s_4326[0], s_4326[1]])
        return g.geometry_length(ls)

    @property
    def distance_steps(self):
        return np.linspace(0, self.distance, num=self.steps)

def _get_index(ygrid, depth):
    import numbers
    if isinstance(depth, numbers.Number):
        return np.argmin(np.abs(ygrid - depth))
    else:
        return [_get_index(ygrid, d) for d in depth]

def invalid_mask(K, ind1=0, ind2=-1, wavelength=0.055):
    from scipy.ndimage import binary_dilation, binary_opening
    from analysis import add_atmospheric
    K = add_atmospheric(K, 0.0, wavelength=wavelength)
    K_last = K[ind1, ind1, ...] + K[ind2, ind2, ...] - 2 * K[ind1, ind2, ...]
    K_last_crop, _ = geospatial.warp(K_last, geospatial_K)
    s1 = np.array(
        [[ 0, 1, 0], [ 1, 1, 1], [0, 1, 0]])
    s2 = np.array(
        [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
    
    invalid = binary_opening(
        binary_dilation(K_last_crop > thresh ** 2, s1), s2, border_value=1)
    return invalid

def read_core_data(fngpkg, geospatial, layer=None):
    import fiona
    import geopandas as gpd
    if layer is None: layer = fiona.listlayers(fngpkg)[0]
    gdf = gpd.read_file(fngpkg, layer=layer).to_crs(geospatial.crs)
    rcs=[]
    for index, row in gdf.iterrows(): # ugly
        rc = tuple(geospatial.rowcol(np.array(row.geometry.xy))[:, 0])
        rcs.append(rc)
    gdf['rowcol'] = rcs
    return gdf

def resample_dem(geospatial, fnraw, fnresampled, upscale=None, overwrite=False):
    from analysis import read_geotiff
    if os.path.exists(fnresampled) and not overwrite:
        return read_geotiff(fnresampled)
    else: 
        dem, _gs = geospatial.warp_from_file(fnraw, upscale=upscale)
        save_geotiff(dem, _gs, fnresampled)
        return dem

def contrast(im, percentiles=(2, 98)):
    for jchannel in range(im.shape[-1]):
        v0, v1 = np.nanpercentile(im[..., jchannel], percentiles)
        im[..., jchannel] = (im[..., jchannel] - v0)/(v1-v0)
    return im

def plot_profile(
        ax, im, geospatial, xy_start, xy_end, steps=512, vlim=None, cmap=None, ymax=None,
        ygrid=None, yticks=None, xticks=None):
    rc = geospatial.rowcol(np.stack((xy_start, xy_end), axis=1))
    pi = ProfileInterpolator(geospatial, xy_start, xy_end, steps=steps)
    profile = pi.interpolate(im)
    # profile_frac = pi.interpolate(frac_thawed)
    if ymax is not None:
        profile = profile[:,:_get_index(ygrid, ymax)]
    vmin, vmax = (0.0, 1.0) if vlim is None else (vlim[0], vlim[1])
    ax.imshow(profile.T, vmin=vmin, vmax=vmax, cmap=cmap, alpha=1)  # profile_frac.T)
    if yticks is not None:
        ax.set_yticks(_get_index(ygrid, yticks))
        ax.set_yticklabels(yticks)
    if xticks is not None:
        ax.set_xticks(_get_index(pi.distance_steps, xticks))
        ax.set_xticklabels(xticks)
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Depth [m]')
    

if __name__ == '__main__':
    from analysis import read_K

    pathres = '/home/simon/Work/gie/processed/kivalina/2019/hadamard/inversion/'
    fnK = '/home/simon/Work/gie/processed/kivalina/2019/hadamard/K_vec.geo.tif'
    fndemraw = '/home/simon/Work/Kivalina/optical/DEM/ArcticDEM/53_19_2_1_2m_v3.0_reg_dem.tif'
    fndemres = os.path.join(pathres, 'DEM.tif')
    fnimraw = '/home/simon/Work/Kivalina/optical/Planet/Kivalina2019/20190625_220816_0e26/analytic_sr_udm2/20190625_220816_0e26_3B_AnalyticMS_SR.tif'
    fnimres = os.path.join(pathres, 'optical.tif')
    fngpkg = '/home/simon/Work/Kivalina/geology/cores2005.gpkg'
    upscale = 16
    wavelength, thresh = 0.055, 4e-3
    nodata = -1
    
    # ir = InversionResults.from_file(os.path.join(pathres, 'ir.p'))
    # geospatial = ir.geospatial
    # ygrid = ir.ygrid
    # save_object(geospatial, os.path.join(pathres, 'geospatial.p'))
    geospatial = load_object(os.path.join(pathres, 'geospatial.p'))
    ygrid = np.arange(0, 1.5, step=2e-3)


    K, geospatial_K = read_K(fnK)
    invalid = invalid_mask(K, ind1=4, wavelength=wavelength)

    xy_start, xy_end = (-164.7450, 67.8441), (-164.7276, 67.8508)
    # xy_start, xy_end = (-164.8153, 67.8571), (-164.7870, 67.8573)

    dem = resample_dem(geospatial, fndemraw, fndemres, upscale=upscale)
    optical = resample_dem(geospatial, fnimraw, fnimres, upscale=upscale)
    e_mean = np.load(os.path.join(pathres, 'e_mean.npy'))
    frac_thawed = np.load(os.path.join(pathres, 'frac_thawed_None.npy'))
    
    cores = read_core_data(fngpkg, geospatial)
    
    
    from scripts.plotting import prepare_figure
    import colorcet as cc
    import matplotlib.pyplot as plt
    cmap = cc.cm['bmy']
    elim = (0.0, 0.5)
    yticks = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    fig, axs = prepare_figure(
        nrows=2, ncols=3, figsize=(2, 0.8), left=0.02, right=0.99, sharex=False, 
        sharey=False)
    ys = [(0.05, 0.15), (0.20, 0.30), (0.40, 0.50)]
    
    for jy, y in enumerate(ys):
        _e_mean = np.mean(
            e_mean[..., _get_index(ygrid, y[0]):_get_index(ygrid, y[1])], axis=-1)
        _e_mean[invalid] = np.nan
        ax = axs[0, jy]
        ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
        ax.set_facecolor('#dddddd')
    cols = ['w', 'k']
    c = [cols[int(x)] for x in cores.code]
    ax.scatter([x[1] for x in cores.rowcol], [x[0] for x in cores.rowcol], s=4, edgecolors=c, linewidths=0.5, c='none')
    ax.set_xlim(axs[0, 0].get_xlim())
    ax.set_ylim(axs[0, 0].get_ylim())
    axs[1, 0].imshow(dem[0, ...], cmap=cc.cm['CET_L10'])
    axs[1, 1].imshow(contrast(np.moveaxis(optical[0:3, ...], 0, -1)))
    plot_profile(
        axs[1, 2], e_mean, geospatial, xy_start, xy_end, steps=512, ymax=0.5, vlim=elim, 
        ygrid=ygrid, cmap=cmap, yticks=yticks)
    plt.show()
    
    
    e_mean_ = np.mean(e_mean[..., _get_index(ygrid, 0.40):_get_index(ygrid, 0.50)], axis=-1) # 0.5
    e_mean_[invalid] = nodata
    # print(np.nanpercentile(e_mean_, (10, 25, 50, 75, 90)))


    vmax = 0.5
    xticks = (0, 500, 1000)

    
    # profile_frac = profile_frac[:,:_get_index(ygrid, ymax)]
    
    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(profile.T, vmin=0.0, vmax=vmax, cmap=cmap, alpha=1)  # profile_frac.T)
    # ax.set_yticks(_get_index(ygrid, yticks))
    # ax.set_yticklabels(yticks)
    # ax.set_xticks(_get_index(pi.distance_steps, xticks))
    # ax.set_xticklabels(xticks)
    # ax.set_xlabel('Distance [m]')
    # ax.set_ylabel('Depth [m]')
    # ax.set_ylims()
    # # plt.imshow(e_mean_)
    # plt.show()
    save_geotiff(e_mean_[np.newaxis, ...], geospatial, os.path.join(pathres, 'e_mean.tif'), nodata=nodata)

