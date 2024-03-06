'''
Created on Mar 30, 2023

@author: simon
'''
import numpy as np
import shutil

from analysis.ioput import load_object, enforce_directory

def export_results(pathin, pathout):
    fnmeta = pathout / 'meta.txt'
    geospatial = load_object(pathin / 'geospatial.p')
    ygrid = load_object(pathin / 'ygrid.p')
    enforce_directory(fnmeta)
    with open(fnmeta, 'w') as f:
        f.writelines((repr(geospatial), '\n', f'ygrid: {ygrid}'))
    for ft in ('e_mean', 'e_quantile'):
        shutil.copy(pathin / f'{ft}.npy', pathout)

if __name__ == '__main__':
    from pathlib import Path

    rmethod = 'hadamard'

    pathout0 = Path('/home/simon/Work/gie/shared')

    path0 = Path('/home/simon/Work/gie/processed/Dalton_131_363/')
    years = ('2019', '2022')
    stacks = ('icecut', 'happyvalley')
    for year in years:
        for stack in stacks:
            path1 = Path(f'{path0}/{stack}/{year}/{rmethod}')
            pathout = Path(f'{pathout0}/{stack}/{year}')
            export_results(path1, pathout)

    year = 2019
    stack = 'kivalina'
    path0 = Path('/home/simon/Work/gie/processed')
    path1 = Path(f'{path0}/{stack}/{year}/{rmethod}')
    pathout = Path(f'{pathout0}/{stack}/{year}')
    export_results(path1, pathout)
