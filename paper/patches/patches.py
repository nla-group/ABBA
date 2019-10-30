import numpy as np
import sys
sys.path.append('./../..')
from ABBA import ABBA
import matplotlib.pyplot as plt
from util import myfigure

ts = np.sin(np.linspace(0, 40, 2000))
ts -= np.mean(ts)
ts /= np.std(ts)

abba = ABBA(tol=0.1, scl=0, min_k=3, max_k=10, verbose=1)
pieces = abba.compress(ts)
APLA = abba.inverse_compress(ts[0], pieces)

string, centers = abba.digitize(pieces)

reconstructed_ts = abba.inverse_transform(string, centers, ts[0])

ABBA_patches = abba.get_patches(ts, pieces, string, centers)
# Construct mean of each patch
d = {}
for key in ABBA_patches:
    d[key] = list(np.mean(ABBA_patches[key], axis=0))

# Stitch patches together
patched_ts = np.array([ts[0]])
for letter in string:
    patch = d[letter]
    patch -= patch[0] - patched_ts[-1] # shift vertically
    patched_ts = np.hstack((patched_ts, patch[1:]))

# Plot SAX representation
fig, (ax1, ax2) = myfigure(nrows=2, ncols=1, fig_ratio=0.71, fig_scale=1)
ax1.plot(ts, color='k', label='original time series')
ax1.plot(APLA, '--', label='reconstruction after compression')
ax1.plot(reconstructed_ts, ':', label='reconstruction after digitization')
ax1.legend()

print('Symbolic representation:', string)

ax2.plot(ts[0:250], color='k',  label='original time series')
next(ax2._get_lines.prop_cycler)
ax2.plot(reconstructed_ts[0:250], ':', label='standard reconstruction')
ax2.plot(patched_ts[0:250], '-.', label='patched reconstruction')
ax2.legend()

plt.savefig('patches.pdf', dpi=300, transparent=True)
