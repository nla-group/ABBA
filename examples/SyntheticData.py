import sys
sys.path.append('./..')
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from ABBA import ABBA


# Noise
#-----------------------------------------------------------------------------#
ts = np.random.randn(1, 100)[0]
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)
abba = ABBA()
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])
plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1, 'r')
plt.legend(['original', 'ABBA'], fontsize=18)
plt.title('Noise', fontsize=18)
plt.show()


# Flat with spikes
#-----------------------------------------------------------------------------#
ts = [0]*100
ts[11] = 1
ts[25] = 1
ts[45] = 1
ts[87] = 1
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)
abba = ABBA()
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])
plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1, 'r')
plt.legend(['original', 'ABBA'], fontsize=18)
plt.title('Flat with spikes', fontsize=18)
plt.show()


# Sine waves
#-----------------------------------------------------------------------------#
ts = np.sin(np.arange(0,73,.1))
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)
abba = ABBA()
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])
plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1, 'r')
plt.legend(['original', 'ABBA'], fontsize=18)
plt.title('Sine wave', fontsize=18)
plt.show()


# Sum of sine waves
#-----------------------------------------------------------------------------#
ts = 0.6*np.sin(np.arange(0,73,.1)) + 0.3*np.sin(0.8*np.arange(0,73,.1))
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)
abba = ABBA()
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])
plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1, 'r')
plt.legend(['original', 'ABBA'], fontsize=18)
plt.title('Sum of sine wave', fontsize=18)
plt.show()


# Stitched linear pieces
#-----------------------------------------------------------------------------#
t1 = np.arange(0, 10, 0.5)
t2 = np.arange(10, 11, 0.1)
t3 = np.arange(11, 6, -0.5)
t4 = np.arange(6, 20, 1)
t5 = np.arange(20, 0, -1)
t6 = np.arange(0, 1, 0.2)
t7 = np.arange(1, 4, 0.5)
t8 = np.arange(4, 0, -0.1)
ts = np.hstack([t1, t2, t3, t4, t5, t6, t7])
ts = ts + 0.5*np.random.randn(len(ts))
abba = ABBA()
string, centers = abba.transform(ts)
ts1 = abba.inverse_transform(string, centers, ts[0])
plt.figure(figsize=(10,7), dpi=80, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(ts, 'k')
plt.plot(ts1, 'r')
plt.legend(['original', 'ABBA'], fontsize=18)
plt.title('Stitched linear pieces', fontsize=18)
plt.show()
