'''
Created on Nov 23, 2019

@author: simon
'''

from pathlib import Path
import socket

hostname = socket.gethostname()

if hostname == 'Vienna':
    path00 = Path('/10TBstorage/Work/')
elif hostname == 'homer':
    path00 = Path('/home2/Work')
else:
    path00 = Path.home() / 'Work/'

path0 = path00 / 'gie'
pstacks = path00 / 'stacks' if hostname in ('Vienna', 'Homer') else path0 / 'stacks'

paths = {'simulation': path0 / 'simulation',
         'stacks': pstacks,
         'processed': path0 /  'processed',
         'forcing': path0 /  'forcing',
         'figures': path0 /  'figures',
         'cores': path0 /  'cores',
         'ancillary': path0 /  'ancillary'}