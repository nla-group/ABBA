"""
Example illustrates:
    - cluster names ordered by most occuring
"""

import sys
sys.path.append('./..')
from ABBA import ABBA
import numpy as np
import collections

tolerance = 0.1
x = np.linspace(0,5000,5001)
ts = np.sin(0.005*np.pi*x)

abba = ABBA(tol=tolerance, min_k=2, max_k=10, scl=0)
string, centers = abba.transform(ts)
counter = collections.Counter(string)
print(counter)
