'''
Created on Nov 23, 2019

@author: simon
'''

import os
import socket

hostname = socket.gethostname()

if hostname == 'Vienna':
    path0 = '/10TBstorage/Work/gie/'

else:
    path0 = os.path.join(os.path.expanduser('~'), 'Work/gie/')


paths = {'simulation': os.path.join(path0, 'simulation'),
         'stacks': os.path.join(path0, 'stacks'),
         'processed': os.path.join(path0, 'processed'),
         'forcing': os.path.join(path0, 'forcing')}