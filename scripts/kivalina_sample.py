'''
Created on Jul 24, 2023

@author: simon
'''
import os
import numpy as np

from analysis import load_object
from scripts.kivalina_index_comparison import geospatial_proc, _read_config, path0



def sample_points(shape, N, max_distance=4, rng=None):
    # max_distance in pixels
    md2 = max_distance ** 2
    if rng is None:
        rng = np.random.default_rng()
    n, rows, cols = 0, [], []
    while n < N:
        candidate = rng.integers(0, shape[0]*shape[1])
        r, c = np.unravel_index(candidate, shape)
        dist2 = (np.array(rows) - r) ** 2 + (np.array(cols) - c) ** 2
        if len(dist2) == 0 or np.min(dist2) > md2:
            rows.append(r)
            cols.append(c)
            n += 1
        print(n)
    return [(r, c) for r, c in zip(rows, cols)]
    

if __name__ == '__main__':
    fncovariates = '/home/simon/Work/gie/ancillary/GEE/covariates.tif'
    fnout = '/home/simon/Work/gie/ancillary/GEE/sample.csv'
    
    ft = load_object(os.path.join(path0, '2019r', 'forcing_timing.p'))
    indranges_names = ft['indranges_names']
    config = ('2019r', 'TDD900_lastday')
    ebar = _read_config(config, indranges_names, geospatial_proc, path0)['mean']
    import rasterio
    src = rasterio.open(fncovariates)
    covariates = geospatial_proc.warp_from_file(fncovariates)[0]
    img = np.concatenate((ebar[np.newaxis, ...], covariates))
    names = ('ebar',) + src.descriptions            

    rng = np.random.default_rng(123697)
    N = 15000
    max_distance = 2.5
    shape = img.shape[1:]
    samples = sample_points(shape, N, rng=rng, max_distance=max_distance)
    import pandas as pd
    df = pd.DataFrame([img[:, r, c] for r, c in samples], columns=names)
    df.to_csv(fnout)
    