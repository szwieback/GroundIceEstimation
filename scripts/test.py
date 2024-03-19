from pathlib import Path
import numpy as np



from GroundIceEstimation.analysis import load_object, InversionResultsMmap, MulticlassInversionResultsMmap
from GroundIceEstimation.analysis.inversion import Mmap
path0 = Path('/home/simon/Work/gie/processed/kivalina/index_ecotype/')
path1 = path0 / '2019'
ft = load_object(path1 / 'forcing_timing.p')
indranges_names = ft['indranges_names']
# e_mean = np.load(path1 / 'e_mean.npy')
e_mean_p= np.load(path1 / 'e_mean_period_mean.npy')
# yf_mean = np.load(path1 / 'yf_mean.npy')
ind = 2 #TDD900_lastday
import matplotlib.pyplot as plt
plt.imshow(e_mean_p[..., ind], vmin=0.0, vmax=0.5)
# plt.imshow(yf_mean[..., -1])
plt.show()

# irdict = InversionResultsMmap._dict_from_file(path1 / 'ir.p')
# print(irdict.keys())
# lwmmap = Mmap(filename=path1 / 'lwmmap.npy', dtype=irdict['lwmmap'].dtype, shape=irdict['lwmmap'].shape)
# # lw = np.memmap(lwmmap.filename, dtype=lwmmap.dtype, mode='r', shape=lwmmap.shape)
#
# ir = MulticlassInversionResultsMmap(
#     irdict['predens'], lwmmap, ec=irdict['ec'], blocksize=irdict['blocksize'])
# ir0 = ir[0]
# print(ir0.lw.shape)