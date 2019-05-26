"""
Example illustrates:
    - varying the scl parameter, all using kmeans.
"""

import sys
sys.path.append('./..')
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from ABBA import ABBA

t1 = np.arange(0, 10, 0.2)
t2 = np.arange(10, 11, 0.1)
t3 = np.arange(11, 6, -0.5)
t4 = np.arange(6, 20, 1)
t5 = np.arange(20, 0, -1)
t6 = np.arange(0, 1, 0.1)
t7 = np.arange(1, 4, 0.5)
t8 = np.arange(4, 0, -0.1)
ts = np.hstack([t1, t2, t3, t4, t5, t6, t7])
ts = ts + 0.5*np.random.randn(len(ts))
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)


abba = ABBA(tol=0.15, scl=0.25)
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=0.15, scl=0.5)
string, centers = abba.transform(ts)
ts2 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=0.15, scl=1)
string, centers = abba.transform(ts)
ts3 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=0.15, scl=2)
string, centers = abba.transform(ts)
ts4 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=0.15, scl=4)
string, centers = abba.transform(ts)
ts5 = abba.inverse_transform(string, centers, ts[0])


plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1)
plt.plot(ts2)
plt.plot(ts3)
plt.plot(ts4)
plt.plot(ts5)
plt.legend(['original', 'scl=0.25', 'scl=0.5', 'scl=1', 'scl=2', 'scl=4'], fontsize=18)
plt.show()
