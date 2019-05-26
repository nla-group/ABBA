"""
Example illustrates:
    - test max_len to prevent exploding piece
"""

import sys
sys.path.append('./..')
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from ABBA import ABBA


ts = [0]*100
ts[90]= 1
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)

abba = ABBA(tol=[10, 0.01], min_k = 1)
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=[10, 0.01], max_len = 10)
string, centers = abba.transform(ts)
ts2 = abba.inverse_transform(string, centers, ts[0])


plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1)
plt.plot(ts2)
plt.legend(['original', 'maxlen = inf', 'maxlen= 50'], fontsize=18)
plt.show()
