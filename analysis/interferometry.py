'''
Created on Sep 14, 2022

@author: simon
'''

import numpy as np

def add_atmospheric(K, var_atmo, wavelength=0.055, to_length=True):
    if to_length:
        K *= (wavelength / (4 * np.pi)) ** 2
    K_eye = np.eye(K.shape[0])[(Ellipsis,) + (None,) * (len(K.shape) - 2)]
    K += var_atmo * (np.ones_like(K) + K_eye)
    return K