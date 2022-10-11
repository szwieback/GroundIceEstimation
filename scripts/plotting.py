'''
Created on Nov 5, 2021

@author: simon
'''
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
globfigparams = {
    'fontsize':8, 'family':'serif', 'usetex': True,
    'preamble': r'\usepackage{amsmath} \usepackage{times} \usepackage{mathtools}',
    'column_inch':229.8775 / 72.27, 'markersize':24, 'markercolour':'#AA00AA',
    'fontcolour':'#666666', 'tickdirection':'out', 'linewidth': 0.5,
    'ticklength': 2.50, 'minorticklength': 1.1 }

cols = {'true': '#000000', 'est': '#aa9966', 'unc': '#9999ee'}
colslist = ['#324145', '#69818c', '#8c8169', '#aaaaaa']
colslist = ['#2b2d47', '#8a698c', '#b29274', '#aaaaaa']
cmap_e = cc.cm['bmy']

def initialize_matplotlib():
    plt.rc('font', **{'size':globfigparams['fontsize'], 'family':globfigparams['family']})
    plt.rcParams['text.usetex'] = globfigparams['usetex']
    plt.rcParams['text.latex.preamble'] = globfigparams['preamble']
    plt.rcParams['legend.fontsize'] = globfigparams['fontsize']
    plt.rcParams['font.size'] = globfigparams['fontsize']
    plt.rcParams['axes.linewidth'] = globfigparams['linewidth']
    plt.rcParams['axes.labelcolor'] = globfigparams['fontcolour']
    plt.rcParams['axes.edgecolor'] = globfigparams['fontcolour']
    plt.rcParams['xtick.color'] = globfigparams['fontcolour']
    plt.rcParams['xtick.direction'] = globfigparams['tickdirection']
    plt.rcParams['ytick.direction'] = globfigparams['tickdirection']
    plt.rcParams['ytick.color'] = globfigparams['fontcolour']
    plt.rcParams['xtick.major.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.major.width'] = globfigparams['linewidth']
    plt.rcParams['xtick.minor.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.minor.width'] = globfigparams['linewidth']
    plt.rcParams['ytick.major.size'] = globfigparams['ticklength']
    plt.rcParams['xtick.major.size'] = globfigparams['ticklength']
    plt.rcParams['ytick.minor.size'] = globfigparams['minorticklength']
    plt.rcParams['xtick.minor.size'] = globfigparams['minorticklength']
    plt.rcParams['text.color'] = globfigparams['fontcolour']

def prepare_figure(
        nrows=1, ncols=1, figsize=(1.7, 0.8), figsizeunit='col', sharex='col', sharey='row',
        squeeze=True, bottom=0.10, left=0.15, right=0.95, top=0.95, hspace=0.5, wspace=0.1,
        remove_spines=True, gridspec_kw=None, subplot_kw=None):

    initialize_matplotlib()
    if figsizeunit == 'col':
        width = globfigparams['column_inch']
    elif figsizeunit == 'in':
        width = 1.0
    figprops = dict(facecolor='white', figsize=(figsize[0] * width, figsize[1] * width))
    if nrows > 0 and ncols > 0:
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols , sharex=sharex, sharey=sharey, squeeze=squeeze,
            gridspec_kw=gridspec_kw, subplot_kw=subplot_kw)
        plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top, hspace=hspace,
                            wspace=wspace)
    else:
        fig = plt.figure()
        axs = None
    fig.set_facecolor(figprops['facecolor'])
    fig.set_size_inches(figprops['figsize'], forward=True)
    if remove_spines:
        try:
            for ax in axs.flatten():
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
        except:
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)
    return fig, axs

def contrast(im, percentiles=(2, 98)):
    for jchannel in range(im.shape[-1]):
        v0, v1 = np.nanpercentile(im[..., jchannel], percentiles)
        im[..., jchannel] = (im[..., jchannel] - v0) / (v1 - v0)
    return im

def plot_profile(
        ax, im, geospatial, xy_tup, im_frac=None, steps=512, vlim=None, cmap=None,
        c_td=None, ymax=None, ygrid=None, yticks=None, xticks=None, labels=None, 
        x_ylabel=-0.11):
    from analysis import thaw_depth
    xy_start, xy_end = xy_tup
    pi = ProfileInterpolator(geospatial, xy_start, xy_end, steps=steps)
    profile = pi.interpolate(im)
    if ymax is not None:
        profile = profile[:,:_get_index(ygrid, ymax) + 1]
    vmin, vmax = (0.0, 1.0) if vlim is None else (vlim[0], vlim[1])
    if im_frac is None:
        alpha = 1.0
    else:
        profile_frac = pi.interpolate(im_frac)
        alpha = 1.0 #profile_frac.T
        td = thaw_depth(profile_frac, ygrid, return_indices=True)
    ax.imshow(profile.T, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto', alpha=alpha)
    if im_frac is not None:
        if c_td is None: c_td = '#ffffff'
        ax.plot(np.arange(pi.steps), td, c=c_td, lw=0.6, alpha=0.5)
    if yticks is not None:
        ax.set_yticks(_get_index(ygrid, yticks))
        ax.set_yticklabels((np.array(yticks) * 100).astype(np.int64))
    if xticks is not None:
        ax.set_xticks(_get_index(pi.distance_steps, xticks))
        ax.set_xticklabels(xticks)
    if labels is not None:
        for lx, lt in labels:
            ax.text(lx, 0.91, lt, c='w', ha='center', va='baseline', transform=ax.transAxes)
    ax.text(0.50, -0.27, 'distance [m]', va='baseline', ha='center', transform=ax.transAxes)
    ax.text(
        x_ylabel, 0.50, 'depth [cm]', ha='right', va='center', rotation=90,
        transform=ax.transAxes)

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

def add_arrow_line(
        ax, rc, c='#000000', lw=0.7, alpha=1.0, label='', hwidth=70, hlength=90,
        pos_frac=0.55, dlabel=None):
    from matplotlib.patches import FancyArrow
    ax.plot(rc[1,:], rc[0,:], c=c, lw=lw, alpha=alpha)
    size_frac = 0.02

    dx, dy = (rc[1, 0] - rc[1, 1]) * size_frac, (rc[0, 0] - rc[0, 1]) * size_frac
    xm = pos_frac * rc[1, 0] + (1 - pos_frac) * rc[1, 1]
    ym = pos_frac * rc[0, 0] + (1 - pos_frac) * rc[0, 1]
    arrow = FancyArrow(
        xm + dx, ym + dy, -2 * dx, -2 * dy, color=c, width=0, head_width=hwidth,
        head_length=hlength, length_includes_head=False, overhang=0.3,
        zorder=10, linewidth=lw, alpha=alpha)
    if label is not None and label != '':
        if dlabel is None: dlabel = 10 * np.array((dx, dy))
        ax.text(xm + dlabel[0], ym + dlabel[1], label, ha='left', va='center', color=c)
    ax.add_patch(arrow)

def add_scalebar(ax, geospatial, length=500, label=None):
    from matplotlib.lines import Line2D
    hextent = geospatial.extent[0]
    frac = length / hextent
    y = -0.10
    dx = 0.05
    # ax.plot((1 - dx - frac, 1 - dx), (y, y), transform=ax.transAxes)
    line = Line2D(
        (1 - dx - frac, 1 - dx), (y, y), lw=0.8, color='#666666', transform=ax.transAxes)
    line.set_clip_on(False)
    ax.add_line(line)
    if label is not None:
        ax.text(
            1 - dx - frac / 2, 1.6 * y, label, ha='center', va='top',
            transform=ax.transAxes)
