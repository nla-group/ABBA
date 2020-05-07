import numpy as np
import sys
sys.path.append('./../..')
sys.path.append('./..')
import SAX
from ABBA import ABBA
import matplotlib.pyplot as plt
np.random.seed(0)
from util import myfigure

# Construct noisy sine wave with linear trend
linspace = np.linspace(0, 16*2*np.pi, 500)
ts = np.sin(linspace)
ts += 0.2*np.random.randn(500)
ts += np.linspace(0, 4, 500)

x_axis_lim = 7

# Plot SAX representation
fig, (ax1, ax2, ax3, ax4) = myfigure(nrows=4, ncols=1, fig_ratio=0.71, fig_scale=1)
plt.subplots_adjust(left=0.125, bottom=None, right=0.95, top=None, wspace=None, hspace=None)
ts1 = ts
ts1 -= np.mean(ts1)
ts1 /= np.std(ts1)
reduced_ts = SAX.compress(ts1, width = 4)
symbolic_ts = SAX.digitize(reduced_ts, number_of_symbols = 9)
print('SAX representation length:', len(symbolic_ts))
reduced_ts = SAX.reverse_digitize(symbolic_ts, number_of_symbols = 9)
ts_SAX = SAX.reconstruct(reduced_ts, width = 4)
ax1.plot(ts1)
ax1.plot(ts_SAX)
ax1.set(xlim=(0, 500), ylim=(-x_axis_lim, x_axis_lim))
ax1.set_ylabel('(i)', rotation='horizontal', labelpad=20)

# Plot SAX representation of differences time series
ts2 = np.diff(ts)
ts2 -= np.mean(ts2)
ts2 /= np.std(ts2)
reduced_ts = SAX.compress(ts2, width = 4)
symbolic_ts = SAX.digitize(reduced_ts, number_of_symbols = 9)
reduced_ts = SAX.reverse_digitize(symbolic_ts, number_of_symbols = 9)
ts_SAX = SAX.reconstruct(reduced_ts, width = 4)
ax2.plot(ts2)
ax2.plot(ts_SAX)
ax2.set(xlim=(0, 500), ylim=(-x_axis_lim, x_axis_lim))
ax2.set_ylabel('(ii)', rotation='horizontal', labelpad=20)

# Reverse differencing
ax3.plot(ts1)
ax3.plot(np.cumsum(ts_SAX))
ax3.set(xlim=(0, 500), ylim=(-x_axis_lim, x_axis_lim))
ax3.set_ylabel('(iii)', rotation='horizontal', labelpad=20)

# Plot ABBA representation
abba = ABBA(tol=0.5, scl=1, min_k=2, max_k=12, verbose=0)
string, centers = abba.transform(ts1)
print('ABBA representation length:', len(string))
ts_recon = abba.inverse_transform(string, centers, ts1[0])
ax4.plot(ts1)
ax4.plot(ts_recon)
ax4.set(xlim=(0, 500), ylim=(-x_axis_lim, x_axis_lim))
ax4.set_ylabel('(iv)', rotation='horizontal', labelpad=20)

fig.tight_layout()
plt.savefig('linear_trend.pdf')
