"""
Example illustrates:
    - test errors raised
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

try:
    print('Invalid tol:')
    abba = ABBA(tol=(1,2))
    string, centers = abba.transform(ts)
except(ValueError):
    print('error caught \n')

try:
    print('Invalid scl:')
    abba = ABBA(scl=-1)
    string, centers = abba.transform(ts)
except(ValueError):
    print('error caught \n')

try:
    print('Invalid min_k and max_k:')
    abba = ABBA(min_k = 10, max_k = 2)
    string, centers = abba.transform(ts)
except(ValueError):
    print('error caught \n')

try:
    print('Not enough pieces:')
    abba = ABBA(min_k = 100, max_k = 100)
    string, centers = abba.transform(ts)
except(ValueError):
    print('error caught \n')
