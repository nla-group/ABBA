"""
Example illustrates:
    - different verbose outputs
"""

import sys
sys.path.append('./..')
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from ABBA import ABBA

# Construct time series
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

# Normalise
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)

# verbose = 0
print('verbose = 0')
abba = ABBA(tol=0.15, scl=1, min_k=3, max_k=10, verbose=0)
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])

# verbose = 1
print('verbose = 1')
abba = ABBA(tol=0.15, scl=1, min_k=3, max_k=10, verbose=1)
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])

# verbose = 2
print('verbose = 2')
abba = ABBA(tol=0.15, scl=1, min_k=3, max_k=10, verbose=2)
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])
