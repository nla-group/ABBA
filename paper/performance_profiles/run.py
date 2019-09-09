import os
from tslearn.metrics import dtw as dtw
import numpy as np
import sys
sys.path.append('./../..')
sys.path.append('./..')
import SAX
from ABBA import ABBA
import oneD_SAX
import pickle
import warnings
import csv
import matplotlib.pyplot as plt
from mydefaults import mydefaults


class Error(Exception):
   """Base class for other exceptions"""
   pass

class TimeSeriesTooShort(Error):
   """Raised when time series is too short"""
   pass

class NotEnoughPieces(Error):
   """Raised when not enough pieces"""
   pass

class CompressionTolHigh(Error):
   """Raised when not enough pieces"""
   pass

class UnknownError(Error):
   """Raised when not enough unique pieces"""
   # Making assumption Ckmeans is available and suitable tol and scl is given.
   pass


if __name__ == "__main__":
    datadir = './../../../UCRArchive_2018/'
    # tolerances
    tol = [0.05*i for i in range(1,11)]
    # number of symbols
    k = 9
    # scaling parameter
    scl = 0

    # If pickle file does not exist, make it.
    if not os.path.exists('scl'+str(scl)+'.p'):

        # Calculate number of time series, to provide progess information.
        ts_count = 0
        for root, dirs, files in os.walk(datadir):
            for file in files:
                if file.endswith('tsv'):
                    with open(os.path.join(root, file)) as f:
                        content = f.readlines()
                        ts_count += len(content)
        print('Number of time series:', ts_count)

        # Construct list of NaNs
        D_SAX_2 = ts_count*[np.NaN]
        D_oneD_SAX_2 = ts_count*[np.NaN]
        D_ABBA_2 = ts_count*[np.NaN]
        D_SAX_DTW = ts_count*[np.NaN]
        D_oneD_SAX_DTW = ts_count*[np.NaN]
        D_ABBA_DTW = ts_count*[np.NaN]
        D_SAX_2_diff = ts_count*[np.NaN]
        D_oneD_SAX_2_diff = ts_count*[np.NaN]
        D_ABBA_2_diff = ts_count*[np.NaN]
        D_SAX_DTW_diff = ts_count*[np.NaN]
        D_oneD_SAX_DTW_diff = ts_count*[np.NaN]
        D_ABBA_DTW_diff = ts_count*[np.NaN]

        ts_name = ts_count*[''] # time series name for debugging
        compression = ts_count*[np.NaN] # Store amount of compression
        tol_used = ts_count*[np.NaN] # Store tol used
        error = ts_count*[0] # track errors

        # Run through time series
        index = 0
        for root, dirs, files in os.walk(datadir):
            for file in files:
                if file.endswith('tsv'):
                    print('file:', file)

                    # bool to keep track of plots, one plot per classification.
                    need_to_plot = True

                    with open(os.path.join(root, file)) as tsvfile:
                        tsvfile = csv.reader(tsvfile, delimiter='\t')

                        for ind, column in enumerate(tsvfile):
                            ts_name[index] += str(file) + '_' + str(ind) # Save filename + index

                            ts = [float(i) for i in column] # convert to list of floats
                            ts = np.array(ts[1:]) # remove class information

                            # remove NaN from time series
                            ts = ts[~np.isnan(ts)]

                            # Z Normalise the times series
                            norm_ts = (ts -  np.mean(ts))
                            std = np.std(norm_ts, ddof=1)
                            std = std if std > np.finfo(float).eps else 1
                            norm_ts /= std

                            try:
                                # Check length of time timeseries
                                if len(norm_ts) < 100:
                                    raise(TimeSeriesTooShort)

                                # Reset tolerance
                                tol_index = 0

                                # ABBA (Adjust tolerance so at least 20% compression is used)
                                for tol_index in range(len(tol)):
                                    abba = ABBA(tol=tol[tol_index], min_k=k, max_k=k, scl=scl, verbose=0)
                                    pieces = abba.compress(norm_ts)
                                    ABBA_len = len(pieces)
                                    if ABBA_len <= len(norm_ts)/5:
                                        tol_used[index] = tol[tol_index]
                                        break
                                    elif tol_index == len(tol)-1:
                                        raise(CompressionTolHigh)

                                # Check number of pieces
                                if np.size(pieces, 0) < k:
                                    raise(NotEnoughPieces)

                                # will catch min_k issue
                                try:
                                    symbolic_ts, centers = abba.digitize(pieces)
                                except:
                                    raise(UnknownError)

                                ts_ABBA = abba.inverse_transform(symbolic_ts, centers, norm_ts[0])

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


                                # Compute distances
                                D_SAX_2[index] = np.linalg.norm(norm_ts[:len(ts_SAX)] - ts_SAX)
                                D_oneD_SAX_2[index] = np.linalg.norm(norm_ts[:len(ts_oneD_SAX)] - ts_oneD_SAX)
                                D_ABBA_2[index] = np.linalg.norm(norm_ts - ts_ABBA)

                                D_SAX_DTW[index] = dtw(norm_ts[:len(ts_SAX)], ts_SAX)
                                D_oneD_SAX_DTW[index] = dtw(norm_ts[:len(ts_oneD_SAX)], ts_oneD_SAX)
                                D_ABBA_DTW[index] = dtw(norm_ts, ts_ABBA)

                                D_SAX_2_diff[index] = np.linalg.norm(np.diff(norm_ts[:len(ts_SAX)]) - np.diff(ts_SAX))
                                D_oneD_SAX_2_diff[index] = np.linalg.norm(np.diff(norm_ts[:len(ts_oneD_SAX)]) - np.diff(ts_oneD_SAX))
                                D_ABBA_2_diff[index] = np.linalg.norm(np.diff(norm_ts) - np.diff(ts_ABBA))

                                D_SAX_DTW_diff[index] = dtw(np.diff(norm_ts[:len(ts_SAX)]), np.diff(ts_SAX))
                                D_oneD_SAX_DTW_diff[index] = dtw(np.diff(norm_ts[:len(ts_oneD_SAX)]), np.diff(ts_oneD_SAX))
                                D_ABBA_DTW_diff[index] = dtw(np.diff(norm_ts), np.diff(ts_ABBA))

                                compression[index] = ABBA_len/len(norm_ts) # Store compression amount
                                index += 1

                                if need_to_plot:
                                    fig, ax = plt.subplots(1, 1)
                                    fig, ax = mydefaults(fig, ax, r=0.8)
                                    plt.plot(norm_ts, 'k', label='original')
                                    plt.plot(ts_SAX, '--', label='SAX')
                                    plt.plot(ts_oneD_SAX, '-.', label='1D-SAX')
                                    plt.plot(ts_ABBA, ':', label='ABBA')
                                    plt.legend()
                                    plt.savefig('scl'+str(scl)+'/'+file[0:-4]+'.pdf')
                                    need_to_plot = False

                                print('Progress:', index, '/', ts_count) # print progress

                            except(TimeSeriesTooShort):
                                error[index] = 1
                                compression[index] = np.NaN
                                tol_used[index] = np.NaN
                                index += 1
                                print('Progress:', index, '/', ts_count) # print progress
                                pass

                            except(NotEnoughPieces):
                                error[index] = 2
                                compression[index] = np.NaN
                                tol_used[index] = np.NaN
                                index += 1
                                print('Progress:', index, '/', ts_count) # print progress
                                pass

                            except(CompressionTolHigh):
                                error[index] = 3
                                compression[index] = np.NaN
                                tol_used[index] = np.NaN
                                index += 1
                                print('Progress:', index, '/', ts_count) # print progress
                                pass

                            except(UnknownError):
                                error[index] = 4
                                compression[index] = np.NaN
                                tol_used[index] = np.NaN
                                index += 1
                                print('Progress:', index, '/', ts_count) # print progress
                                pass

        D = {}
        D['ts_name'] = ts_name
        D['compression'] = compression
        D['tol_used'] = tol_used
        D['error'] = error
        D['k'] = k
        D['scl'] = scl
        D['tol'] = tol

        D['SAX_2'] = D_SAX_2
        D['oneD_SAX_2'] = D_oneD_SAX_2
        D['ABBA_2'] = D_ABBA_2

        D['SAX_DTW'] = D_SAX_DTW
        D['oneD_SAX_DTW'] = D_oneD_SAX_DTW
        D['ABBA_DTW'] = D_ABBA_DTW

        D['SAX_2_diff'] = D_SAX_2_diff
        D['oneD_SAX_2_diff'] = D_oneD_SAX_2_diff
        D['ABBA_2_diff'] = D_ABBA_2_diff

        D['SAX_DTW_diff'] = D_SAX_DTW_diff
        D['oneD_SAX_DTW_diff'] = D_oneD_SAX_DTW_diff
        D['ABBA_DTW_diff'] = D_ABBA_DTW_diff

        with open('scl'+str(scl)+'.p', 'wb') as handle:
            pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
