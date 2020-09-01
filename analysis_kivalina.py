'''
Created on Nov 29, 2019

@author: simon
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from InterferometricSpeckle.storage import load_object, save_object, prepare_figure

from phase_reference import displacement_time_series
from pathnames import paths

model_inference = 'GaussianBandedInversePlusRankOneU3'
stackname = 'kivalina'

def plot_kivalina(pathout, fnout=None, overwrite=False):
    dispdicts = {}
    for locn, loc in locations.items():
        fn = os.path.join(pathout, f'disp_{locn}.p')
        if not os.path.exists(fn) or overwrite:
            dispdict = displacement_time_series(
                location=loc, stackname=stackname, location_reference=location_reference,
                paths=paths)
            save_object(dispdict, fn)
        else:
            dispdict = load_object(fn)
        dispdicts[locn] = dispdict

    fig, axs = prepare_figure(ncols=2, sharex=True, sharey=True, figsize=(0.7, 0.3),
                              left=0.03)
    for jp, (jloc, locn) in enumerate(
        zip([0, 1, 0], ['ridge', 'colonizedfloodplain', 'polygons'])):
        disp = dispdicts[locn]['disp']
        xpos = np.arange(len(disp)) + 1 - (jp - 1) * 0.12
    #     plt.plot(xpos, disp, marker='o')
        C = dispdicts[locn]['C'] + np.eye(len(xpos)) * 9e-6  # 3 mm atmosphere error
        disperr = np.sqrt(np.diag(C))
        colj = '#507c26'
        if locn == 'ridge':
            colj = '#b1c89b'
        axs[jloc].plot(xpos, disp, lw=1.0, c=colj, zorder=2)
        axs[jloc].errorbar(xpos, disp, yerr=disperr, lw=0.0, elinewidth=0.5, c=colj,
                     ecolor=colj, marker='o', markersize=2, zorder=3)
        print(xpos)
    axs[0].set_yticks([-0.1, 0])
    axs[0].set_yticklabels([])
    axs[0].set_xticks([2.05, 4.65, 7.10])  # 1 July, 1 August, 1 September
    axs[0].set_xticklabels([])
#     plt.show()
    if fnout is not None:
        plt.savefig(fnout, transparent=True)

def plot_kivalina_location(
        locationname, location, year, pathout, location_reference=None, overwrite=True):
    fn = os.path.join(pathout, f'disp_{locationname}.p')
    stackname_ = stackname + str(year)
    fnout = os.path.join(pathout, f'{locationname}.pdf')
    if not os.path.exists(fn) or overwrite:
        dispdict = displacement_time_series(
            location=location, stackname=stackname_, location_reference=location_reference,
            paths=paths, overwrite=overwrite)
        save_object(dispdict, fn)
    else:
        dispdict = load_object(fn)

    fig, ax = prepare_figure(ncols=1, sharex=True, sharey=True, figsize=(0.32, 0.3),
                              left=0.05, right=0.99)
    disp = dispdict['disp']
    xpos = np.arange(len(disp)) + 1
#     plt.plot(xpos, disp, marker='o')
    C = dispdict['C'] + np.eye(len(xpos)) * 9e-6  # 3 mm atmosphere error
    disperr = np.sqrt(np.diag(C))
    colj = '#507c26'
    ax.plot(xpos, disp, lw=1.0, c=colj, zorder=2)
    ax.errorbar(xpos, disp, yerr=disperr, lw=0.0, elinewidth=0.5, c=colj,
                 ecolor=colj, marker='o', markersize=2, zorder=3)

    ax.set_yticks([-0.1, 0])
    ax.set_yticklabels([])
    ax.set_xticks([2.05, 4.65, 7.10])  # 1 July, 1 August, 1 September
    ax.set_xticklabels([])
    ax.set_ylim([-0.17, 0.0])
#     plt.show()
    if fnout is not None:
        plt.savefig(fnout, transparent=True)

if __name__ == '__main__':
    from InterferometricSpeckle.probability import covariance_matrix_from_sample, ProcessedOptimizationStack
    from InterferometricSpeckle.storage import filename
    year = 2019
    stackname_ = stackname + str(year)

#     fields_out = {'batch': 'processed', 'L': 100, 'phase-closure': True,
#                   'cohcorr': 'ccpr'}
#     fnstack = filename(fields=fields_out, meta=None, category='processed',
#                      subcategories=(stackname_,), paths=paths)
#     pospc = ProcessedOptimizationStack.from_file(fnstack)
#     phi = pospc.by_model('phases', model_name=model_inference, is_short_name=False)
#     y = pospc.as_array(pospc.y)
#     C_obs = covariance_matrix_from_sample(y)
#     plt.imshow(10 * np.log10(np.abs(C_obs[..., 3, 3])))
#     coh = np.abs(C_obs[..., 5, 6] / np.sqrt(C_obs[..., 5, 5] * C_obs[..., 6, 6]))
# #     plt.imshow(coh, vmin=0, vmax=1)
#     plt.show()
#     raise
    locations = {'polygons': (93, 153), 'colonizedfloodplain': (110, 143),
                 'ridge': (72, 183), 'hillslope': (94, 175), 'polygons2': (105, 172)}
    location_reference = {2019: (103, 157), 2018: (104, 159), 2017: (105, 159)}


    #88, 207
    #82, 191 ridge; 2017: 83, 19
    # 100, 166 polyg
    # 99, 162 also polyg
    # 100, 
    # 110, 143; floodplain
    for year in (2017, 2018, 2019):
        stackname_ = stackname + str(year)
        dts = displacement_time_series(
        location=(105, 172), location_reference=location_reference[year], stackname=stackname_,
        paths=paths, overwrite=False)
        print(dts['disp'])
    

#     for year in (2017, 2019):
#         for location in ('polygons2',):
#             pathout = os.path.join(paths['processed'], stackname + str(year), 'timeseries')
#             plot_kivalina_location(
#                 location, locations[location], year, pathout, 
#                 location_reference=location_reference[year], overwrite=True)

