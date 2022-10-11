'''
Created on Oct 5, 2022

@author: simon
'''
import numpy as np
import os
from analysis import Geospatial, save_geotiff, load_object, save_object, InversionResults
from scripts.kivalina_analysis import resample_dem
year = '2019'#'2022'
pathres = f'/home/simon/Work/gie/processed/Dalton_131_363/icecut/{year}/hadamard'
fnimraw = '/home/simon/Work/gie/ancillary/Planet/20220620/20220620_211420_05_249d/analytic_sr_udm2/20220620_211420_05_249d_3B_AnalyticMS_SR.tif'
fnimres = os.path.join(pathres, 'optical.tif')
upscale = 8
site = np.array((-148.8317, 69.0414))[:, np.newaxis]

ir = InversionResults.from_file(os.path.join(pathres, 'ir.p'))
geospatial = ir.geospatial
ygrid = ir.ygrid
save_object(geospatial, os.path.join(pathres, 'geospatial.p'))
geospatial = load_object(os.path.join(pathres, 'geospatial.p'))
ygrid = np.arange(0, 1.5, step=2e-3)
e_mean = np.load(os.path.join(pathres, 'e_mean.npy'))
e_quantile = np.load(os.path.join(pathres, 'e_quantile.npy'))
frac_thawed = np.load(os.path.join(pathres, 'frac_thawed_None.npy'))
rc_site = geospatial.rowcol(site)
# rc_site[0,0] -=1
e_mean_site = e_mean[rc_site[0, 0], rc_site[1, 0],:]

e_quantile_site = e_quantile[rc_site[0, 0], rc_site[1, 0], ...]
frac_site = frac_thawed[rc_site[0, 0], rc_site[1, 0]]
print(ygrid[np.nonzero(frac_site < 1 / 2)[0][0]])

import matplotlib.pyplot as plt
from scripts.plotting import prepare_figure, cmap_e, colslist, _get_index, contrast
fig, ax = prepare_figure(nrows=1, ncols=1)
ax.fill_betweenx(ygrid, e_quantile_site[:, 0], e_quantile_site[:, 1], edgecolor='none', facecolor=colslist[0], alpha=0.07)
ax.plot(e_mean_site, ygrid, c=colslist[0])
ax.set_ylim((0.60, 0))
plt.show()

cmap = cmap_e
elim = (0.0, 0.5)
xticks_im = (25, 65, 105, 145)
yticks_im = (10, 50)
ys = [(0.05, 0.15), (0.20, 0.30), (0.50, 0.60)]
fig, axs = prepare_figure(ncols=3, nrows=2, sharex='none', sharey='none')

optical = resample_dem(geospatial, fnimraw, fnimres, upscale=upscale)

for jy, y in enumerate(ys):
    _e_mean = np.mean(
        e_mean[..., _get_index(ygrid, y[0]):_get_index(ygrid, y[1])], axis=-1)
    # _e_mean[invalid] = np.nan
    ax = axs[0, jy]
    im_e = ax.imshow(_e_mean, cmap=cmap, vmin=elim[0], vmax=elim[1])
    ax.set_facecolor('#aaaaaa')
    ax.set_xticks(xticks_im)
    ax.set_yticks(yticks_im)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='#dddddd', linewidth=0.4)
optical = optical[::-1, ...][0:3]
ax = axs[1][0]
ax.imshow(contrast(np.moveaxis(optical, 0, -1)))
plt.show()
