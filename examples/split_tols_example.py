"""
Example illustrates:
    - using different tols for compression and digitisation
"""

import sys
sys.path.append('./..')
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from ABBA import ABBA


ts = np.random.normal(size = 701)
ts = np.cumsum(ts)
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)

compression_tol = 0.1
digitisation_tol = 0.4

tol = [compression_tol, digitisation_tol]
abba = ABBA(tol=tol)
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])

plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1)
plt.legend(['original', 'ABBA'], fontsize=18)
plt.show()
