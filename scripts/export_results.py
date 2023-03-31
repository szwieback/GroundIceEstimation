'''
Created on Mar 30, 2023

@author: simon
'''
import os, shutil
import numpy as np

from analysis.ioput import load_object, enforce_directory

def export_results(pathin, pathout):
    fnmeta = os.path.join(pathout, 'meta.txt')
    geospatial = load_object(os.path.join(pathin, 'geospatial.p'))
    ygrid = load_object(os.path.join(pathin, 'ygrid.p'))
    enforce_directory(fnmeta)
    with open(fnmeta, 'w') as f:
        f.writelines((repr(geospatial), '\n', f'ygrid: {ygrid}'))
    for ft in ('e_mean', 'e_quantile'):
        shutil.copy(os.path.join(pathin, f'{ft}.npy'), pathout)    

if __name__ == '__main__':
    rmethod = 'hadamard'

    pathout0 = '/home/simon/Work/gie/shared'

    path0 = '/home/simon/Work/gie/processed/Dalton_131_363/'    
    years = ('2019', '2022')
    stacks = ('icecut', 'happyvalley')
    for year in years:
        for stack in stacks:
            path1 = f'{path0}/{stack}/{year}/{rmethod}'
            pathout = f'{pathout0}/{stack}/{year}'
            export_results(path1, pathout)

    year = 2019
    stack = 'kivalina'
    path0 = '/home/simon/Work/gie/processed'
    path1 = f'{path0}/{stack}/{year}/{rmethod}'    
    pathout = f'{pathout0}/{stack}/{year}'    
    export_results(path1, pathout)
