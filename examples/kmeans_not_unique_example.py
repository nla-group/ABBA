"""
Example illustrates:
    - demonstrate kmeans suboptimal convergence (ocassionaly shown in this example).
      Currenlty unable to find example where demonstarted every time.
"""

import sys
sys.path.append('./..')
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from ABBA import ABBA

ts = np.sin(np.arange(0,100,.1))
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)



abba = ABBA(tol=0.01, scl=0.0001)
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=0.01, scl=0.0001)
string, centers = abba.transform(ts)
ts2 = abba.inverse_transform(string, centers, ts[0])


plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1)
plt.plot(ts2)
plt.legend(['original', 'ABBA', 'ABBA'], fontsize=18)
plt.show()
