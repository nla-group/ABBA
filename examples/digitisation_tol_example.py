"""
Example illustrates:
    - setting digitisation_tol = 0
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


abba = ABBA(tol=[0.15,0], max_k=10)
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=[0.15,0], max_k=8)
string, centers = abba.transform(ts)
ts2 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=[0.15,0], max_k=6)
string, centers = abba.transform(ts)
ts3 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=[0.15,0], max_k=4)
string, centers = abba.transform(ts)
ts4 = abba.inverse_transform(string, centers, ts[0])

abba = ABBA(tol=[0.15,0], max_k=2)
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
plt.legend(['original', '10 symbols', '8 symbols', '6 symbols', '4 symbols', '2 symbols'], fontsize=18)
plt.show()
