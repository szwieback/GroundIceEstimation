'''
Created on Nov 28, 2019

@author: simon
'''
'''
Created on Nov 23, 2019

@author: simon
'''
import os

from InterferometricSpeckle.storage import load_object
from InterferometricSpeckle.probability import (cohcorr_phase_rmse_library, 
                                                GaussianBandedInversePlusRank1Speckle)
from InterferometricSpeckle.dataanalysis import process_stack


from pathnames import paths

def phaselinking_kivalina(year=2017, overwrite=False):
    stackname = f'kivalina{year}'
    fnstack = os.path.join(paths['stacks'], stackname + '.npy')
    stack = load_object(fnstack)

    phase_closure = True
    cohcorr = cohcorr_phase_rmse_library[100]
    specklem_inference = (
        GaussianBandedInversePlusRank1Speckle(
                    phase_closure=phase_closure, cohcorr=cohcorr, U=2),
        GaussianBandedInversePlusRank1Speckle(
                    phase_closure=phase_closure, cohcorr=cohcorr, U=3),)
    process_stack(stack, stackname=stackname, batch_size=50, n_jobs=-1,
                  phase_closure=phase_closure, cohcorr=cohcorr, paths=paths,
                  specklem_inference=specklem_inference, overwrite=overwrite)

if __name__ == '__main__':
    phaselinking_kivalina(year=2017)
    phaselinking_kivalina(year=2018)
