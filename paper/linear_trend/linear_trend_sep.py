import numpy as np
import sys
sys.path.append('./../..')
sys.path.append('./..')
import SAX
from ABBA import ABBA
import matplotlib.pyplot as plt
np.random.seed(0)
from mydefaults import mydefaults

# Construct noisy sine wave with linear trend
linspace = np.linspace(0, 16*2*np.pi, 500)
ts = np.sin(linspace)
ts += 0.2*np.random.randn(500)
ts += np.linspace(0, 4, 500)

# Plot SAX representation
fig, ax = plt.subplots(1)
fig, ax = mydefaults(fig, ax, r=0.25, s=0.6)
ts1 = ts
ts1 -= np.mean(ts1)
ts1 /= np.std(ts1)
reduced_ts = SAX.compress(ts1, width = 4)
symbolic_ts = SAX.digitize(reduced_ts, number_of_symbols = 9)
print('SAX representation length:', len(symbolic_ts))
reduced_ts = SAX.reverse_digitize(symbolic_ts, number_of_symbols = 9)
ts_SAX = SAX.reconstruct(reduced_ts, width = 4)
ax.plot(ts1)
ax.plot(ts_SAX)
plt.savefig('SAX.pdf')

# Plot SAX representation of differences time series
fig, ax = plt.subplots(1)
fig, ax = mydefaults(fig, ax, r=0.25, s=0.6)
ts2 = np.diff(ts)
ts2 -= np.mean(ts2)
ts2 /= np.std(ts2)
reduced_ts = SAX.compress(ts2, width = 4)
symbolic_ts = SAX.digitize(reduced_ts, number_of_symbols = 9)
reduced_ts = SAX.reverse_digitize(symbolic_ts, number_of_symbols = 9)
ts_SAX = SAX.reconstruct(reduced_ts, width = 4)
ax.plot(ts2)
ax.plot(ts_SAX)
plt.savefig('SAX_diff.pdf')

# Reverse differencing
fig, ax = plt.subplots(1)
fig, ax = mydefaults(fig, ax, r=0.25, s=0.6)
ax.plot(ts1)
ax.plot(np.cumsum(ts_SAX))
plt.savefig('cumsum_SAX_diff.pdf')

# Plot ABBA representation
fig, ax = plt.subplots(1)
fig, ax = mydefaults(fig, ax, r=0.25, s=0.6)
abba = ABBA(tol=0.5, scl=1, min_k=2, max_k=12, verbose=0)
string, centers = abba.transform(ts1)
print('ABBA representation length:', len(string))
ts_recon = abba.inverse_transform(string, centers, ts1[0])
ax.plot(ts1)
ax.plot(ts_recon)
plt.savefig('ABBA.pdf')
