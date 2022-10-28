'''
Created on Oct 27, 2022

@author: simon
'''
# PYPROJ_GLOBAL_CONTEXT=ON
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from scripts.plotting import prepare_figure, contrast, colslist
from analysis import read_geotiff, Geospatial
os.environ['PYPROJ_GLOBAL_CONTEXT'] = 'ON'

bbox = [69.02, 69.17, -148.92, -148.64]

def map_alaska(ax):

    crs_pc = ccrs.PlateCarree()
    ax.set_facecolor('#dddddd')
    ax.set_extent([-168, -144, 64.8, 71.0], crs=crs_pc)

    ax.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'land', '50m', edgecolor='#666666', facecolor='#eeeeee', linewidth=0.2))

    latlon = (67.727222, -164.539167)
    ax.scatter(latlon[1], latlon[0], 7, transform=crs_pc, zorder=3, c='#ffffff')
    ax.scatter(latlon[1], latlon[0], 4, transform=crs_pc, zorder=3, c=colslist[1])
    ax.text(
        latlon[1] + 1.4, latlon[0], 'TI', va='center', transform=crs_pc,
        c=colslist[1])

    rect = plt.Rectangle(
        (bbox[2], bbox[0]), bbox[3] - bbox[2], bbox[1] - bbox[0], ec=colslist[0],
        fc=colslist[0], linewidth=2, transform=ccrs.Geodetic(), zorder=10)

    ax.add_patch(rect)

def load_landsat_ard(path0, scene, bands):
    def _fn(band):
        fn = f'{scene}_SR_B{band}.TIF'
        return os.path.join(path0, scene, fn)
    im = np.concatenate([read_geotiff(_fn(band)) for band in bands])
    geospatial = Geospatial.from_file(_fn(bands[0]))
    return im, geospatial

def map_dalton(fnout=None):
    fig, ax = prepare_figure(
        nrows=1, ncols=1, figsize=(2.45, 3.60), figsizeunit='in', left=0.01, right=0.99, 
        top=0.995, bottom=0.005)
    from scripts.pathnames import paths
    path0 = os.path.join(paths['ancillary'], 'Landsat')
    scene = 'LC08_AK_016002_20200703_20210504_02'
    im, geospatial = load_landsat_ard(path0, scene, [4, 3, 2])

    bbox = [69.02, 69.17, -148.92, -148.64]
    dlat = -2e-4  # 4
    geospatial_crop = Geospatial.plate_carree(bbox, dlat=dlat)

    im_crop, _geospatial_crop = geospatial_crop.warp(im, geospatial)
    im_crop = np.moveaxis(im_crop, 0, -1).astype(np.float64)
    im_cc = contrast(im_crop)
    plt.imshow(im_cc, origin='upper')

    # show IC, HV bbox
    def add_rectangle(ll, ur, label=None):
        r, c = geospatial_crop._rc_bbox(ll, ur)
        print(r, c)
        rect = plt.Rectangle(
            (c[0], r[0]), c[1] - c[0], r[1] - r[0], ec=colslist[0], fc='none',
            linewidth=1.4, zorder=10)
        ax.add_patch(rect)
        if label is not None:
            tbox = dict(
                facecolor=colslist[0], alpha=0.8, edgecolor='none',
                boxstyle='round, pad=0.3')
            ax.text(
                np.mean(c), np.mean(r), label, ha='center', va='center', zorder=10,
                c='#ffffff', bbox=tbox)
    from scripts.happyvalley import ll, ur
    add_rectangle(ll, ur, label='Happy Valley')
    from scripts.icecut import ll, ur
    add_rectangle(ll, ur, label='Ice Cut')

    # labels
    ax.text(0.20, 0.54, 'road', c='#ffffff', rotation=-60, transform=ax.transAxes)

    # show inset
    crs_ak = ccrs.LambertAzimuthalEqualArea(-151, 63)
    # isax = fig.add_axes([0.70, 0.83, 0.28, 0.18], projection=crs_ak, frameon=False)
    isax = fig.add_axes([0.700, -0.017, 0.28, 0.18], projection=crs_ak, frameon=False)
    map_alaska(isax)
    
    # show scale
    from scripts.plotting import add_scalebar
    add_scalebar(
        ax, geospatial_crop, length=1000, y=0.95, dx=0.024, lw=1.4, color='#ffffff', 
        label='1 km', ylab=0.985)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout)

if __name__ == '__main__':
    from scripts.pathnames import paths
    fnout = os.path.join(paths['figures'], 'akmap.pdf')
    map_dalton(fnout=fnout)

    # map_alaska()

