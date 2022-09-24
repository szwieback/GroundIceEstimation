'''
Created on Sep 7, 2022

@author: simon
'''
import numpy as np
import pandas as pd
import datetime
import os

from analysis import StefanPredictor, PredictionEnsemble, enforce_directory
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple,
    StefanStratigraphyConstantE)

def read_merra_subset(fn, field='T2MMEAN[0][0]'):
    with open(fn, 'r') as f:
        dstr = f.readline().split('.')[-2]
        d = datetime.datetime.strptime(dstr, '%Y%m%d')
        parts = [p.strip() for p in f.readline().split(', ')]
        assert len(parts) == 2
        assert parts[0] == field
        T = float(parts[1])
    return d, T

def load_forcing_merra_subset(folder, to_Celsius=True):
    ld = os.listdir(folder)
    def _match(fn1, fn2, comps=(0, 1, 3, 4)):
        p1, p2 = fn1.split('.'), fn2.split('.')
        return all([p1[co] == p2[co] for co in comps])
    fn0 = ld[0]
    fns = [fn for fn in ld if _match(fn, fn0)]
    vals = [read_merra_subset(os.path.join(folder, fn)) for fn in fns]
    df = pd.DataFrame(vals, columns=('datetime', 'T'))
    df.sort_values(by='datetime', inplace=True)
    df = df.set_index('datetime')
    if to_Celsius:
        df['T'] = df['T'] - 273.15
    return df

def parse_dates(datestr, strp='%Y%m%d'):
    if isinstance(datestr, str):
        return datetime.datetime.strptime(datestr, strp)
    else:
        return [parse_dates(ds, strp=strp) for ds in datestr]

def kivalina_forcing(folder_forcing, year=2019):
    df = load_forcing_merra_subset(folder_forcing)
    d0 = {2019: '2019-05-10', 2017: '2017-05-10', 2018: '2018-05-10'}[year]
    d1 = {2019: '2019-09-15', 2017: '2017-09-20', 2018: '2018-09-15'}[year]
    d0_, d1_ = parse_dates((d0, d1), strp='%Y-%m-%d')
    dailytemp = (df.resample('D').mean())['T'][pd.date_range(start=d0, end=d1)]
    dailytemp[dailytemp < 0] = 0
    datesstr = {2019:
             ('20190606', '20190618', '20190630', '20190712', '20190724', '20190805',
              '20190817', '20190829', '20190910')}
    datesdisp = [datetime.datetime.strptime(d, '%Y%m%d') for d in datesstr[year]]
    ind_scenes = [int((d - d0_).days) for d in datesdisp]
    return dailytemp, ind_scenes

def process_kivalina(year=2019, rmethod='hadamard'):
    path0 = f'/10TBstorage/Work/stacks/Kivalina/gie/{year}/proc/{rmethod}/geocoded'
    folder_forcing = '/10TBstorage/Work/gie/forcing/kivalina'
    pathout = f'/10TBstorage/Work/gie/processed/kivalina/{year}/{rmethod}'
    # path0 = f'/home/simon/Work/gie/processed/kivalina/{year}/{rmethod}'
    # folder_forcing = '/home/simon/Work/gie/forcing/Kivalina'
    # pathout = os.path.join(path0, 'temp')
    geom = {'ia': 39.29 / 180 * np.pi}
    wavelength = 0.055
    var_atmo = (3e-3) ** 2
    xy_ref = np.array([-164.7300, 67.8586])[:, np.newaxis]
    # xy_ref = np.array([-164.79600, 67.8700])[:, np.newaxis]
    # ll, ur = (-164.8660, 67.8400), (-164.7185, 67.8600)
    ll, ur = (-164.8175, 67.8370), (-164.7185, 67.8600)
    
    N = 10000
    Nbatch = 1

    from analysis import (
        read_K, add_atmospheric, read_referenced_motion, InversionProcessor,
        InversionResults)

    fnunw = os.path.join(path0, 'unwrapped.geo.tif')
    fnK = os.path.join(path0, 'K_vec.geo.tif')
    K, geospatial_K = read_K(fnK)
    K = add_atmospheric(K, var_atmo)
    s_obs, geospatial = read_referenced_motion(fnunw, xy=xy_ref, wavelength=wavelength)
    assert geospatial == geospatial_K
    from analysis.ioput import save_geotiff
    
    save_geotiff(s_obs - s_obs[4, ...][np.newaxis, ...], geospatial, os.path.join(pathout, 's_obs_late.tif'))
    
    dailytemp, ind_scenes = kivalina_forcing(folder_forcing, year=year)
    
    predictor = StefanPredictor()
    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)

    
    data = {'s_obs': s_obs, 'K': K}
    for dname in data.keys():
        data[dname], geospatial_crop = geospatial.crop(data[dname], ll=ll, ur=ur)
    ip = InversionProcessor(predens, geospatial=geospatial_crop)
    ir = ip.results(
        ind_scenes, data['s_obs'], data['K'], pathout=pathout, n_jobs=-1, overwrite=True)
    ir.save(os.path.join(pathout, 'ir.p'))
    ir = InversionResults.from_file(os.path.join(pathout, 'ir.p'))
    
    expecs = [
        ('e', 'mean'), ('e', 'var'), ('yf', 'mean'), ('s_los', 'mean'),
        ('s_los', 'var'), ('frac_thawed', None, {'ind_scene': ind_scenes[-1]}),
        ('e', 'quantile', {'quantiles': (0.1, 0.9)})]
    for expec in expecs:
        kwargs = expec[2] if len(expec) == 3 else {}
        ir.export_expectation(pathout, param=expec[0], etype=expec[1], **kwargs)

if __name__ == '__main__':
    process_kivalina()

'''
    from stackpro import (
        S1IsceReader, BoxcarMultilooker, DDStack, SnaphuUnwrapper, MLPhaseLinker,
        GregPhaseLinker, GdalGeocoder)
    from greg import assemble_tril, covariance_matrix, valid_G, regularize_G, EMI, correlation
    import matplotlib.pyplot as plt

    fnstack = '/home/simon/Work/gie/processed/kivalina/2019/none/stack.ddp'
    fnC = '/home/simon/Work/gie/processed/kivalina/2019/C.npy'

#     stack = DDStack.from_file(fnstack)
#     print(stack['slc'].shape)
#     C_raw = covariance_matrix(np.moveaxis(stack['slc'].astype(np.complex128), -2, -1))
#     C_obs = correlation(C_raw)
#     print(C_obs.shape)
#     np.save(fnC, C_obs)
    rtype = 'hadamard'
    C_obs = np.load(fnC)
    C_obs = C_obs[20:120, 200:300, ...]
    G0 = valid_G(C_obs, corr=True)
    G = regularize_G(G0, rtype=rtype, alpha=0.8, nu=0.8)
    cphases = EMI(C_obs, G=G, corr=False)
    phases = np.angle(cphases * cphases[..., 6].conj()[..., np.newaxis])[..., :]
    print(phases.shape)

    plt.imshow(phases[..., -1])
    plt.show()
#     Cp = C_obs[97, 175, ...]
# #     Cp = C_obs[97, 100, ...]
# #     plt.imshow(np.abs(Cp))
# #     plt.imshow(np.abs(C_obs[..., 0, 6]), vmin=0, vmax=1)
# #     plt.imshow(np.angle(C_obs[..., 0, 5]), vmin=-np.pi, vmax=np.pi)
#     plt.imshow(stack['phases'][..., 3])
#     plt.show()


'''
