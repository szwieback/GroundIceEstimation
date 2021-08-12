'''
Created on Aug 9, 2021

@author: simon
'''

import os
import numpy as np
import pickle
import zlib

def enforce_directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass
        
def save_object(obj, filename):
    enforce_directory(os.path.dirname(filename))
    if os.path.splitext(filename)[1].strip() == '.npy':
        np.save(filename, obj)
    else:
        with open(filename, 'wb') as f:
            f.write(zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))

def load_object(filename):
    if os.path.splitext(filename)[1].strip() == '.npy':
        return np.load(filename)
    with open(filename, 'rb') as f:
        obj = pickle.loads(zlib.decompress(f.read()))
    return obj

