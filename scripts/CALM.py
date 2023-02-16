'''
Created on Dec 4, 2022

@author: simon
'''
import datetime as dt
import numpy as np

def read_CALM_csv(fn, sep=';'):
    datadict = {}
    def extract_date_year(entries):
        dates, years = [], []
        for entry in entries:
            dstr = entry.split('-')[1]
            year2 = int(dstr[-2:])
            year = 2000 + year2 if year2 < 50 else 1900 + year2
            day = int(dstr[-4:-2])
            month = int(dstr[:-4])
            dates.append(dt.date(year, month, day))
            years.append(year)
        return dates, np.array(years)
    with open(fn, 'r') as f:
        for jl, l in enumerate(f.readlines()):
            entries = l.strip().split(sep)
            if jl == 0:
                datadict['date'], datadict['year'] = extract_date_year(entries[1:])
            else:
                datadict[entries[0]] = np.array([float(x) for x in entries[1:]])
    return datadict

def plot_CALM_HV(fn_hv, year_min=None, xticks=None, fnout=None):
    from plotting import prepare_figure, colslist
    import matplotlib.pyplot as plt
    fig, ax = prepare_figure(
        figsize=(2.1, 1.3), figsizeunit='in', bottom=0.23, left=0.18, right=0.98, top=0.96)
    datadict = read_CALM_csv(fn_hv)
    if year_min is None: year_min = min(datadict['year'])
    years, means = (datadict[k][datadict['year'] >= year_min] for k in ('year', 'mean'))
    ax.plot(years, means, lw=0.7, c=colslist[0], alpha=0.7)
    hlyears = (2019, 2022)
    ms = 4
    mc = colslist[0]
    for year in years:
        if year in hlyears:
            ax.plot(
                year, means[years == year], linestyle='none', marker='o', c=mc, ms=ms)
        else:
            ax.plot(
                year, means[years == year], linestyle='none', marker='o', c=mc, mfc='w', ms=ms)
    ylim = (50, 35)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylim(ylim)
    ax.text(
        -0.16, 0.50, 'thaw depth [cm]', va='center', ha='right', transform=ax.transAxes, rotation=90)
    ax.text(0.50, -0.29, 'year', ha='center', va='baseline', transform=ax.transAxes)    
    if fnout is not None:
        fig.savefig(fnout)
    else:
        plt.show()

if __name__ == '__main__':
    from pathnames import paths
    import os
    fn_hv = os.path.join(paths['ancillary'], 'CALM', 'U9b_alt_2007_2022.csv')
    fnout = os.path.join(paths['figures'], 'CALM.pdf')
    plot_CALM_HV(fn_hv, year_min=2013, xticks=[2013, 2016, 2019, 2022], fnout=fnout)

