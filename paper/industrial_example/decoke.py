import pickle
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
from tslearn.metrics import dtw as dtw

import sys
sys.path.append('./../..')
from ABBA import ABBA
from util import myfigure

tolerance = 0.1
ts = pickle.load(open('decoke.p', 'rb'))

abba = ABBA(tol=tolerance, min_k=10, max_k=30)
string, centres = abba.transform(ts)

ts_new = np.array(abba.inverse_transform(string, centres, ts[0]))

pieces = abba.compress(ts)
ts_con = abba.inverse_compress(ts[0], pieces)

print('np.linalg.norm(ts_con-ts) =', np.linalg.norm(ts_con-ts))
print('should be bounded by = ', np.sqrt(len(ts))*tolerance)
print('dtw(ts_con - ts_new) = ', dtw(ts_con, ts_new))
print('dtw(ts - ts_new) = ', dtw(ts, ts_new))

# Plot ABBA representation
fig, ax = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1.6)
plt.plot(ts, color='k')
plt.plot(ts_con, color='#D95319', linestyle='--')
plt.plot(ts_new, color='#0072BD', linestyle=':')
plt.xlim([1, 7200])
plt.ylim([-2.3, 2.3])
plt.legend(['original time series', 'reconstruction after compression', 'reconstruction after digitization'], loc=9)
plt.savefig('decoke_'+str(abba.tol)+'_scl'+str(abba.scl)+'.pdf')


# Construct list of vertical errors
v_error = np.zeros((len(string), 1))
for ind, letter in enumerate(string):
    pc = centres[ord(letter)-97,:]
    v_error[ind] = pieces[ind][1]-pc[1]

fig, ax = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1.6)
for i in range(50):
    plt.plot(np.arange(.5,len(v_error)+.5,1),np.cumsum(np.random.permutation(v_error)), color='#D9D9D9')
#plt.plot(np.cumsum(v_error), color='#0072BD', label='reconstruction error')
plt.plot(np.arange(.5,len(v_error)+.5,1),np.cumsum(v_error), color='#0072BD', label='reconstruction error')

p1 = np.arange(0, len(v_error)+1)
p2 = np.arange(len(v_error), -1, -1)

inc_std = np.std(pieces[:,1])
tol_s = np.sqrt(6/len(pieces))*(tolerance/0.2)
var_bound = np.multiply(p1, (p2/len(v_error)))*tol_s*tol_s
bound = np.sqrt(var_bound)

plt.plot(np.arange(0, len(v_error)+1),bound, color = '#D95319', label='bound')
plt.plot(np.arange(0, len(v_error)+1),-bound, color = '#D95319')
plt.ylim([-0.9, 0.9])
plt.legend(ncol=2, loc=8)

plt.savefig('brownian_bridge.pdf')
