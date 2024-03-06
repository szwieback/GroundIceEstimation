'''
Created on Nov 23, 2019

@author: simon
'''

from pathlib import Path
import socket

hostname = socket.gethostname()

if hostname == 'Vienna':
    path0 = Path('/10TBstorage/Work/gie/')
elif hostname == 'homer':
    path0 = Path('/home2/Work/gie')
else:
    path0 = Path.home() / 'Work/gie/'


paths = {'simulation': path0 / 'simulation',
         'stacks': path0 / 'stacks',
         'processed': path0 /  'processed',
         'forcing': path0 /  'forcing',
         'figures': path0 /  'figures',
         'cores': path0 /  'cores',
         'ancillary': path0 /  'ancillary'}