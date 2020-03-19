import os
from tslearn.metrics import dtw as dtw
import numpy as np
import sys
sys.path.append('./../../src')
import SAX
from ABBA import ABBA
import oneD_SAX
import pickle
import csv
import matplotlib.pyplot as plt


# tolerance
tol = 0.03
# tol step size
tol_step = 0.01
# number of symbols
k = 9
# scaling parameter
scl = 0


# Calculate number of time series, to provide progess information.
datadir = './../../../UCRArchive_2018/'

# Construct list of all files
file_list = []
for root, dirs, files in os.walk(datadir):
    for file in files:
        if file != 'README.md': # ignore README.md files
            file_list.append((root, file))

# Choose random element
index = np.random.randint(0, len(file_list))
(root, file) = file_list[index]

# Construct list of time series
timeseries = []
with open(os.path.join(root, file)) as tsvfile:
    tsvfile = csv.reader(tsvfile, delimiter='\t')
    for column in tsvfile:
        ts = [float(i) for i in column] # convert to list of floats
        ts = np.array(ts[1:])
        # remove NaN from time series
        ts = ts[~np.isnan(ts)]
        # Z Normalise the times series
        norm_ts = (ts -  np.mean(ts))
        std = np.std(norm_ts, ddof=1)
        std = std if std > np.finfo(float).eps else 1
        norm_ts /= std
        timeseries.append(norm_ts)

# Choose random timeseries
index2 = np.random.randint(0, len(timeseries))
norm_ts = timeseries[index2]

# ABBA (Adjust tolerance so at least 50% compression is used)
while True:
    abba = ABBA(tol=tol, min_k=k, max_k=k, scl=scl, verbose=0)
    pieces = abba.compress(norm_ts)
    ABBA_len = len(pieces)
    if ABBA_len <= len(norm_ts)/2:
        break
    else:
        tol += tol_step

symbolic_ts, centers = abba.digitize(pieces)
ts_ABBA = abba.inverse_transform(symbolic_ts, centers, norm_ts[0])
print('tolerance used:', tol)

# SAX
width = len(norm_ts) // ABBA_len # crop to equal number of segments as ABBA.
reduced_ts = SAX.compress(norm_ts[0:width*ABBA_len], width = width)
symbolic_ts = SAX.digitize(reduced_ts, number_of_symbols = k)
SAX_len = len(symbolic_ts)
reduced_ts = SAX.reverse_digitize(symbolic_ts, number_of_symbols = k)
ts_SAX = SAX.reconstruct(reduced_ts, width = width)

# oneD_SAX
width = max(len(norm_ts) // ABBA_len, 2) # crop to equal number of segments as ABBA.
slope = int(np.ceil(np.sqrt(k)))
intercept = int(np.ceil(np.sqrt(k)))
reduced_ts = oneD_SAX.compress(norm_ts[0:width*ABBA_len], width = width)
symbolic_ts = oneD_SAX.digitize(reduced_ts, width, slope, intercept)
oneD_SAX_len = len(symbolic_ts)
reduced_ts = oneD_SAX.reverse_digitize(symbolic_ts, width, slope, intercept)
ts_oneD_SAX = oneD_SAX.reconstruct(reduced_ts, width = width)

# Plot
plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(norm_ts, 'k', linewidth=2.0)
plt.plot(ts_ABBA, 'b:', linewidth=2.0)
plt.plot(ts_SAX, 'r--', linewidth=2.0)
plt.plot(ts_oneD_SAX, 'r-.', linewidth=2.0)
plt.legend(['original', 'ABBA', 'SAX', '1D-SAX'], fontsize=18)
plt.show()
