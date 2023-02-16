import pandas as pd
import numpy as np
from collections import namedtuple
from collections.abc import Iterable
import os

ESeg = namedtuple('ESeg', ['drange', 'e'])

rho_ice = 917.0
rho_water = 999.8

sitenames = {'HV': 'Happy Valley', 'IC': 'Ice Cut'}

fns = {
    'IC': 'FSA_Dalton_IC_2022_20230202.xlsx', 'HV': 'FSA_Dalton_HV_2022_20230202.xlsx'}

def extract_core(df, method=None):
    data = []
    row = 1
    while row is not None:
        try:
            entry = df.loc[row]
            depth = entry['Depth']
            assert depth > 0
            drange = (entry['Start'], entry['End'])
            facies = entry['Facies'].lower()
            if facies == 'ice':
                e = 1.0
            elif facies == 'missing':
                e = None
            else:
                V_frozen = entry['Volume'] * 1e-6  # to m3
                m_ew = entry['Excess water wt'] * 1e-3  # kg
                V_ew = m_ew / rho_water
                V_ei = (rho_water / rho_ice) * V_ew
                V_thawed = entry['Container only soil'] * 1e-6 # m3
                V_thawedtotal = entry['Container w/water'] * 1e-6 #me              
                if method is None or method == 'supernatant':
                    e = V_ei / V_frozen
                elif method == 'thawed':   # Morse                 
                    e = V_ei / (V_ei + V_thawedtotal) if V_thawed > 0 else 0.0
                elif method == 'difference':
                    e = (V_frozen - V_thawed) / V_frozen if V_thawed > 0 else 0.0
                else:
                    raise ValueError(f"Excess ice method {method} not recognized")
            data.append(ESeg(drange, e))
            row = row + 1
        except:
            row = None
    return data

def interpolate_core(data, delta_y=1, depth=150):  # in cm
    assert delta_y == 1  # for now
    y = np.arange(depth).astype(np.float32)
    e_grid = np.empty_like(y)
    e_grid[:] = np.nan
    for eseg in data:
        if eseg.e is not None:
            e_grid[eseg.drange[0]:eseg.drange[1]] = eseg.e
    return e_grid

def bootstrap_percentiles(d, percentiles=(10, 90), seed=1, size=1000):
    rng = np.random.default_rng(seed)
    d_bs = rng.choice(d, size=(size, d.shape[0]))
    d_mean_bs = np.nanmean(d_bs, axis=1)
    return np.nanpercentile(d_mean_bs, percentiles, axis=0)

def read_site(fn):
    df_dict = pd.read_excel(fn, sheet_name=None, engine='openpyxl')
    data_dict = {core: extract_core(df_dict[core]) for core in df_dict}
    e_grid = np.array([interpolate_core(data_dict[core]) for core in df_dict])
    return e_grid

def plot_sites(fns_abs, fnout=None):
    from scripts.plotting import prepare_figure, colslist
    import matplotlib.pyplot as plt
    fig, axs = prepare_figure(
        ncols=len(fns_abs), figsize=(1, 0.5), sharey=True, bottom=0.21, left=0.12,
        wspace=0.25, sharex=True)
    y_grid = np.arange(150)  # hard-coded for now
    for jsite, site in enumerate(fns_abs.keys()):
        e_grid = read_site(fns_abs[site])
        e_q_mean = bootstrap_percentiles(e_grid, (10, 90))
        e_mean = np.nanmean(e_grid, axis=0)
        axs[jsite].fill_betweenx(
            y_grid, e_q_mean[0,:], e_q_mean[1,:], edgecolor='none', facecolor=colslist[1],
            alpha=0.20)
        axs[jsite].plot(e_grid.T, y_grid, c=colslist[2], alpha=0.16, lw=0.5)
        axs[jsite].plot(e_mean, y_grid, c=colslist[0], lw=1.2)
        axs[jsite].text(
            0.50, 0.98, sitenames[site], ha='center', va='baseline',
            transform=axs[jsite].transAxes)
    axs[0].set_ylim((60, 0))
    axs[0].set_xlim((-0.01, 1.00))
    axs[0].text(
        -0.22, 0.50, '$y$ [cm]', ha='right', va='center', transform=axs[0].transAxes,
        rotation=90)
    for ax in axs: ax.set_xlabel('$e$ [-]')
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def plot_site(fn_abs, fnout=None):
    from scripts.plotting import prepare_figure, colslist
    import matplotlib.pyplot as plt
    fig, ax = prepare_figure(
        figsize=(1.8, 1.3), figsizeunit='in', bottom=0.23, left=0.19, top=0.96, right=0.96)
    y_grid = np.arange(150)  # hard-coded for now

    e_grid = read_site(fn_abs)
    e_q_mean = bootstrap_percentiles(e_grid, (10, 90))
    e_mean = np.nanmean(e_grid, axis=0)
    ax.fill_betweenx(
        y_grid, e_q_mean[0,:], e_q_mean[1,:], edgecolor='none', facecolor=colslist[1],
        alpha=0.20)
    ax.plot(e_grid.T, y_grid, c=colslist[4], alpha=0.15, lw=0.6)
    ax.plot(e_mean, y_grid, c=colslist[0], lw=1.2)
    ax.set_ylim((60, 0))
    ax.set_xlim((-0.01, 1.00))
    ax.text(
        -0.175, 0.500, '$y$ [cm]', ha='right', va='center', transform=ax.transAxes,
        rotation=90)
    ax.text(0.50, -0.29, '$e$ [-]', ha='center', va='baseline', transform=ax.transAxes)
    if fnout is None:
        plt.show()
    else:
        fig.savefig(fnout)

def plot_inset(fnout):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    ccrsproj = ccrs.LambertAzimuthalEqualArea(
        central_longitude=-149, central_latitude=69)
    ccrspc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(1, 0.8), frameon=True, facecolor='#ffffff')
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98), projection=ccrsproj)
    ax.set_extent([-168, -130, 55, 71], crs=ccrspc)

    ax.add_feature(cfeature.OCEAN, color='#dddddd')
    ax.add_feature(cfeature.LAND, color='#ffffff')
    ax.add_feature(cfeature.BORDERS, linestyle='-', lw=0.5, color='#cccccc')
    ax.plot(
        -148.84, 69.15, linestyle='none', marker='o', mfc='#333333', mec='none', ms=3,
        transform=ccrspc)

    ax.spines['geo'].set_linewidth(0.3)
    ax.spines['geo'].set_edgecolor('#666666')
    plt.savefig(fnout, dpi=1200)

if __name__ == '__main__':
    from scripts.pathnames import paths
    fns_abs = {site: os.path.join(paths['cores'], fns[site]) for site in fns}
    # plot_sites(fns_abs, fnout=os.path.join(paths['figures'], 'cores.pdf'))
    for site in fns:
        plot_site(fns_abs[site], fnout=os.path.join(paths['figures'], f'cores_{site}.pdf'))

    df_dict = pd.read_excel(fns_abs['IC'], sheet_name=None, engine='openpyxl')
    data_dict = {core: extract_core(df_dict[core]) for core in df_dict}
    print(data_dict['IC-F'])
#     e_grid = np.array([interpolate_core(data_dict[core]) for core in df_dict])
#     print(np.nanmean(e_grid, axis=0))
