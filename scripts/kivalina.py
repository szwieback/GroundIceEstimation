'''
Created on Sep 7, 2022

@author: simon
'''
import rasterio
import numpy as np
import pandas as pd
import datetime

from greg.preproc import assemble_tril

from analysis import StefanPredictor, InversionSimulator, PredictionEnsemble, load_object
from simulation import (
    StefanStratigraphySmoothingSpline, StratigraphyMultiple,
    StefanStratigraphyConstantE)

class Geospatial():
    def __init__(self, transform, crs, shape=None):
        self.transform = transform
        self.crs = crs
        self.shape = shape

    @classmethod
    def from_file(cls, fn):
        src = rasterio.open(fn)
        shape = (src.height, src.width)
        gsp = Geospatial(transform=src.transform, crs=src.crs, shape=shape)
        del src
        return gsp

    def rowcol(self, xy):
        # xy: lonlat for WGS84
        r, c = rasterio.transform.rowcol(self.transform, xy[0, :], xy[1, :])
        return np.stack((r, c), axis=0)

    def __eq__(self, obj):
        if not isinstance(obj, Geospatial):
            return False
        else:
            eq_transform = (self.transform == obj.transform)
            eq_shape = (self.shape == obj.shape)
            eq_crs = (self.crs == obj.crs)
            return eq_transform and eq_shape and eq_crs

def read_geotiff(fntif):
    src = rasterio.open(fntif)
    arr = src.read()
    del src
    return arr

def read_K(fntif):
    K_vec = read_geotiff(fntif)
    geospatial = Geospatial.from_file(fntif)
    K = np.moveaxis(assemble_tril(np.moveaxis(K_vec, 0, -1)), (0, 1), (-2, -1))
    return K, geospatial

def read_referenced_motion(fnunw, xy=None, wavelength=0.055, flip_sign=True):
    unw = read_geotiff(fnunw)
    unw *= wavelength / (4 * np.pi)
    if xy.shape[1] > 1:
        raise NotImplementedError('Only one reference point')
    geospatial = Geospatial.from_file(fnunw)
    rc = geospatial.rowcol(xy)
    unw_ref = unw[:, rc[0, 0], rc[1, 0]]
    unw -= unw_ref[:, np.newaxis, np.newaxis]
    if flip_sign: unw *= -1
    return unw, geospatial

def add_atmospheric(K, var_atmo, wavelength=0.055, to_length=True):
    if to_length:
        K *= (wavelength / (4 * np.pi)) ** 2
    K_eye = np.eye(K.shape[0])[(Ellipsis,) + (None,) * (len(K.shape) - 2)]
    K += var_atmo * (np.ones_like(K) + K_eye)
    return K

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

if __name__ == '__main__':
    import os
    year = 2019
    path0 = f'/home/simon/Work/gie/processed/kivalina/{year}/hadamard'
    folder_forcing = '/home/simon/Work/gie/forcing/Kivalina'
    fnunw = os.path.join(path0, 'unwrapped.geo.tif')
    fnK = os.path.join(path0, 'K_vec.geo.tif')
    geom = {'ia': 39.29 / 180 * np.pi}
    wavelength = 0.055
    var_atmo = (3e-3) ** 2
    xy_ref = np.array([-164.73101, 67.85766])[:, np.newaxis]
    xy_point = np.array([-164.732488, 67.850668])[:, np.newaxis]

    K, geospatial_K = read_K(fnK)
    K = add_atmospheric(K, var_atmo)
    s_obs, geospatial = read_referenced_motion(fnunw, xy=xy_ref, wavelength=wavelength)
    assert geospatial == geospatial_K

    rc = geospatial.rowcol(xy_point)

    dailytemp, ind_scenes = kivalina_forcing(folder_forcing, year=year)

    N = 10000
    Nbatch = 1
    predictor = StefanPredictor()
    strat = StratigraphyMultiple(
        StefanStratigraphySmoothingSpline(N=N), Nbatch=Nbatch)
    predens = PredictionEnsemble(strat, predictor, geom=geom)
    predens.predict(dailytemp)

    from inference import lw_mvnormal, psislw, expectation, quantile, _normalize
    _K = K[..., rc[0, 0], rc[1, 0]]
    _s_obs = s_obs[..., rc[0, 0], rc[1, 0]]

    rng = np.random.default_rng(seed=1)
    s_pred = predens.extract_predictions(ind_scenes, C_obs=_K, rng=rng)
    lw = lw_mvnormal(_s_obs[np.newaxis, :], _K[np.newaxis, :], s_pred)
    lw_ps, _ = psislw(lw)
    lw_ = _normalize(lw_ps)
    e_mean = expectation(predens.results['e'], lw_, normalize=False)
    import matplotlib.pyplot as plt
    plt.plot(e_mean[0, ...], predens.ygrid)
    print(_s_obs)
    plt.show()
#     print(los[:, rc[0, 0], rc[1, 0]])
#     print(K.shape, los.shape)

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
