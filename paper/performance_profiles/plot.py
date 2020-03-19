################################################################################
# Generate performance plots from data in given pickle file.

filename = 'scl0.p'

################################################################################

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PerformanceProfile import PerformanceProfile as pp
import math
import csv
import sys
sys.path.append('./../..')
from ABBA import ABBA

D = pickle.load(open(filename, 'rb'))
measures = ['2', 'DTW', '2_diff', 'DTW_diff']

# Check if directory exists
if not os.path.exists(filename[0:-2]):
    os.mkdir(filename[0:-2])

# Save figures in folder
for m in measures:
    plt.close()
    P = (np.vstack([D['SAX_'+m], D['oneD_SAX_'+m], D['ABBA_'+m]])).T
    P = P[~np.isnan(P).any(axis=1)] # remove NaN rows
    pp(P, 10, file_name=filename[0:-2] + '/performance/' + m + '.pdf', alg_legend=['SAX', '1d-SAX', 'ABBA'], markevery=5)

# If txt file exists, delete it
if os.path.exists(filename[0:-2] + '/info.txt'):
  os.remove(filename[0:-2] + '/info.txt')

# Create text file with key information and save in same folder
with open(filename[0:-2] + '/performance/info.txt', 'a') as f:
    compression = np.array(D['compression'])
    ind = ~np.isnan(compression)
    failures = np.sum(np.isnan(compression))
    f.write('k: '+str(D['k'])+'\n')
    f.write('scl: '+str(D['scl'])+'\n')
    form_tol = [ '%.2f' % elem for elem in D['tol'] ]
    f.write('tol: '+str(form_tol)+'\n'+'\n')
    f.write('Number of time series: '+str(len(D['compression']))+'\n')
    f.write('Success: '+ str(len(D['compression'])-failures)+'\n')
    f.write('Failures: '+str(failures)+'\n'+'\n')
    f.write('Average compression percentage: '+str(np.mean(compression[ind]))+'\n'+'\n')
    d = {}
    d['nan'] = 0
    for i in D['tol_used']:
        if math.isnan(i):
            d['nan'] += 1
        elif i in d:
            d[i] += 1
        else:
            d[i] = 1
    f.write('tol used: \n')
    f.write('nan: '+str(d['nan'])+'\n')
    del d['nan'] # sort not support < between float and string
    for i in sorted(d):
        f.write(str(i)[0:4]+': '+str(d[i])+'\n')

    f.write('\n')
    f.write('No error: '+ str(D['error'].count(0))+'\n')
    f.write('Time series too short: '+ str(D['error'].count(1))+'\n')
    f.write('Not enough pieces: '+ str(D['error'].count(2))+'\n')
    f.write('Data too noisy: '+ str(D['error'].count(3))+'\n')
    f.write('Unknown error: '+ str(D['error'].count(4))+'\n')
