import sys
sys.path.append('./..')
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from ABBA import ABBA
from tslearn.metrics import dtw as dtw
import random
import os
import csv

print('This example requires the UCR Classififcation Archive in the same directory as the ABBA module folder!')

datadir = './../../UCRArchive_2018/'
folder_list = ['InlineSkate', 'InspectWingbeatSound', 'LargeKitchenAppliances',
                'Adiac', 'StarLightCurves', 'Fungi', 'MelbournePedestrian',
                'GestureMidAirD1', 'SmoothSubspace', 'PigCVP', 'EOGHorizontalSignal',
                'Phoneme', 'Plane', 'ACSF1']


# Select some ts from UCR Suite
# ---------------------------------------------------------------------------- #
ts_list = []
for folder in folder_list:
    for root, dirs, files in os.walk(datadir+folder):
        for file in files:
            if file.endswith('TRAIN.tsv'):
                with open(os.path.join(root, file)) as tsvfile:
                    tsvfile = csv.reader(tsvfile, delimiter='\t')
                    col = next(tsvfile)
                    ts = [float(i) for i in col] # convert to list of floats
                    ts = np.array(ts[1:]) # remove class information

                    # remove NaN from time series
                    ts = ts[~np.isnan(ts)]

                    # Z Normalise the times series
                    norm_ts = (ts -  np.mean(ts))
                    std = np.std(norm_ts)
                    std = std if std > np.finfo(float).eps else 1
                    norm_ts /= std

                ts_list.append(norm_ts)


# Plot the ts
# ---------------------------------------------------------------------------- #
plt.figure()
for i in ts_list:
    plt.plot(i)
plt.legend(folder_list)
plt.show()


# perform test
# ---------------------------------------------------------------------------- #
abba = ABBA(verbose=0)

for ind, ts in enumerate(ts_list):
    pieces = abba.compress(ts)
    compressed_ts = abba.inverse_compress(ts[0], pieces)
    string, centers = abba.digitize(pieces)
    print(centers)

    pieces = abba.inverse_digitize(string, centers)
    pieces = abba.quantize(pieces)
    symbolic_ts = abba.inverse_compress(ts[0], pieces)

    if ind == 0:
        L = ['index', 'N', 'n', 'euclid(T,hat{T})', 'k', 'dtw(hat{T},tilde{T})', 'tol', 'sqrt{N}tol']
        frmt = "{:>6}{:>6}{:>6}{:>25}{:>4}{:>25}{:>6}{:>20}"
        print(frmt.format(*L))

    L = [ind, len(ts), len(string), np.linalg.norm(ts - compressed_ts), len(set(string)), dtw(compressed_ts, symbolic_ts), abba.tol, np.sqrt(len(ts))*abba.tol]
    frmt = "{:>6}{:>6}{:>6}{:>25}{:>4}{:>25}{:>6}{:>20}"
    print(frmt.format(*L))
