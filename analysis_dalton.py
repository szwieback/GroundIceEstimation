'''
Created on Nov 27, 2019

@author: simon
'''
import matplotlib.pyplot as plt
import numpy as np
import os

from InterferometricSpeckle.storage import load_object, save_object, prepare_figure

from phase_reference import displacement_time_series
from pathnames import paths

model_inference = 'GaussianBandedInversePlusRankOneU2'
wavelength = 0.056



if __name__ == '__main__':
    locations = {'polygons': (62, 189), 'colonizedfloodplain': (99, 175)}
    # (101, 171)
    stackname = 'dalton'
    col = '#507c26'

    overwrite = False
    pathout = os.path.join(paths['processed'], stackname, 'timeseries')
    dispdicts = {}
    for locn, loc in locations.items():
        fn = os.path.join(pathout, f'disp_{locn}.p')
        if not os.path.exists(fn) or overwrite:
            dispdict = displacement_time_series(location=loc, stackname=stackname, 
                                                paths=paths)
            save_object(dispdict, fn)
        else:
            dispdict = load_object(fn)
        dispdicts[locn] = dispdict
    fig, axs = prepare_figure(ncols=2, sharex=True, sharey=True, figsize=(0.7, 0.3),
                              left=0.03)
    for jloc, locn in enumerate(['polygons', 'colonizedfloodplain']):
        disp = dispdicts[locn]['disp']
        xpos = np.arange(len(disp)) + 1
    #     plt.plot(xpos, disp, marker='o')
        C = dispdicts[locn]['C'] + np.eye(len(xpos)) * 1e-6  # 1 mm atmosphere error
        disperr = np.sqrt(np.diag(C))
        axs[jloc].errorbar(xpos, disp, yerr=disperr, lw=1.0, elinewidth=0.5, c=col,
                           ecolor=col, marker='o', markersize=2)
    axs[0].set_yticks([-0.1, 0])
    axs[0].set_yticklabels([])
    axs[0].set_xticks([1.67, 4.0, 6.5])  # 1 July, 1 August, 1 September
    axs[0].set_xticklabels([])
#     plt.show()
    plt.savefig(os.path.join(pathout, 'displacements_raw_dalton.pdf'), transparent=True)
