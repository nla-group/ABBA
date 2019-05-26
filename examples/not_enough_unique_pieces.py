"""
Example illustrates:
    - Resorting to kmeans as not enough unique pieces for Ckmeans
"""

import sys
sys.path.append('./..')
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from ABBA import ABBA

abba = ABBA(scl = 0, min_k = 5)

pieces = np.array([[1.0, 1, 1],[1, 2, 1],[0, 1, 0],[2, 3, 4],[4, -1, 4],[0, 1, 0]])
string, centers = abba.digitize(pieces)

print(centers)
