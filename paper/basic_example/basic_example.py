import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
np.random.seed(0)

import sys
sys.path.append('./../..')
from ABBA import ABBA
from util import myfigure

# Construct time series
t1 = np.arange(0, 10, 0.5)              # up
t2 = 9.5*np.ones(50)                    # flat
t3 = 10.5*np.ones(50)                    # flat
t4 = np.arange(10, 20, 0.5)             # up
t5 = np.arange(20, 0, -1)               # down
t6 = np.arange(0, 10, 0.5)              # up
t7 = 10*np.ones(50)                     # flat
ts = np.hstack([t1, t2, t3, t4, t5, t6, t7])
ts = ts + 0.5*np.random.randn(len(ts))

# Normalise
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)
print('Length of time series:', len(ts))


# Plot compression
fig, ax = myfigure(nrows=1, ncols=1, fig_ratio=0.5, fig_scale=1)
ax.axis([0, 230, -3, 3])
plt.xlabel('time point')
plt.ylabel('value')
ax.plot(ts, color='k', linestyle='-')
plt.tight_layout()
plt.savefig('time_series.pdf', dpi=300, transparent=True)

abba = ABBA(tol=0.4, scl=0, min_k=3, max_k=10, verbose=1)
pieces = abba.compress(ts)
APLA = abba.inverse_compress(ts[0], pieces)

# Digitization
string, centers = abba.digitize(pieces)
print(string)

# Plot compression
fig, ax = myfigure(nrows=1, ncols=1, fig_ratio=0.8, fig_scale=1.7)

ax.axis([0, 230, -3, 4.5])
plt.xlabel('time point')
plt.ylabel('value')
ax.plot(ts, color='k', linestyle='-', label='original time series')
ax.plot(APLA, color='#D95319', linestyle='--', label='polygonal chain approximation')
ax.annotate('1', (pieces[0,0]/2-8, ts[0]+pieces[0,1]/2))
ax.annotate('2', (pieces[0,0]+pieces[1,0]/2, ts[0]+pieces[0,1]+pieces[1,1]/2+0.2))
ax.annotate('3', (np.sum(pieces[0:2,0])+pieces[2,0]/2, ts[0]+np.sum(pieces[0:2,1])+pieces[2,1]/2+0.2))
ax.annotate('4', (np.sum(pieces[0:3,0])+pieces[3,0]/2-8, ts[0]+np.sum(pieces[0:3,1])+pieces[3,1]/2))
ax.annotate('5', (np.sum(pieces[0:4,0])+pieces[4,0]/2+5, ts[0]+np.sum(pieces[0:4,1])+pieces[4,1]/2))
ax.annotate('6', (np.sum(pieces[0:5,0])+pieces[5,0]/2+5, ts[0]+np.sum(pieces[0:5,1])+pieces[5,1]/2))
ax.annotate('7', (np.sum(pieces[0:6,0])+pieces[6,0]/2, ts[0]+np.sum(pieces[0:6,1])+pieces[6,1]/2+0.2))
plt.legend()
plt.tight_layout()
plt.savefig('compression.pdf', dpi=300, transparent=True)
plt.savefig('compression.png', dpi=300, transparent=True)
#plt.close()


# Define colormap
c3 = (0.2,0.678,1)
c1 = (0.968,0.867,0.631)
c2 = (0.961,0.737,0.639)

# Plot digitization with scl=0
fig, ax = myfigure(nrows=1, ncols=1, fig_ratio=0.8, fig_scale=1.7)
ax.axis([0, 80, -8, 6])
plt.xlabel('length')
plt.ylabel('increment')
split1 = (centers[1,1] + centers[2,1])/2
split2 = (centers[0,1] + centers[1,1])/2
len = 81
#ax.fill_between(np.arange(0, len), [6]*len, [split2]*len, facecolor='none', hatch=2*'X', edgecolor=(0.961,0.737,0.639), linewidth=0.0)
#ax.fill_between(np.arange(0, len), [split2]*len, [split1]*len, facecolor='none', linewidth=0.0)
#ax.fill_between(np.arange(0, len), [split1]*len, [-8]*len, facecolor='none', hatch=2*'+', edgecolor=(0.6,0.839,1), linewidth=0.0)
ax.fill_between(np.arange(0, len), [6]*len, [split2]*len, facecolor=c2, linewidth=0.0)
ax.fill_between(np.arange(0, len), [split2]*len, [split1]*len, facecolor=c1, linewidth=0.0)
ax.fill_between(np.arange(0, len), [split1]*len, [-8]*len, facecolor=c3, linewidth=0.0)
plt.scatter(pieces[:,0], pieces[:,1], marker='x', c='black', s = 20, label='pieces')
plt.scatter(centers[:,0], centers[:,1], marker='o', c='red', s = 15, label='cluster centers')
plt.text(70, centers[0,1], 'a', ha='center', fontweight='bold', wrap=True)
plt.text(70, centers[1,1], 'b', ha='center', fontweight='bold', wrap=True)
plt.text(70, centers[2,1], 'c', ha='center', fontweight='bold', wrap=True)
for i in range(pieces.shape[0]):
    ax.annotate(str(i+1), (pieces[i,0]-2, pieces[i,1]+0.2))
plt.legend(loc=1)
plt.tight_layout()
plt.savefig('digitization0.pdf', dpi=300, transparent=True)
plt.savefig('digitization0.png', dpi=300, transparent=True)

# Plot digitisation with scl=1
abba = ABBA(tol=0.4, scl=1, min_k=3, max_k=10, verbose=1)
string, centers = abba.transform(ts)
print(string)
# Construct mesh
x_min, x_max = 0, 80
y_min, y_max = -8, 6
xx, yy = np.meshgrid(np.arange(x_min, x_max, .2), np.arange(y_min, y_max, .2))
cmap = colors.ListedColormap([c1, c2, c3])
bounds = [-1,0.5,1.5,3]
norm = colors.BoundaryNorm(bounds, cmap.N)
mesh = np.c_[xx.ravel(), yy.ravel()]
Z = []
for index, entry in enumerate(mesh):
    Z.append(np.argmin(np.sum(np.abs(centers-entry)**2,axis=-1)**(1./2)))
Z = np.array(Z).reshape(xx.shape)

fig, ax = myfigure(nrows=1, ncols=1, fig_ratio=0.8, fig_scale=1.7)
plt.xlabel('length')
plt.ylabel('increment')
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=cmap, norm=norm,
           aspect='auto', origin='lower')
plt.scatter(pieces[:,0], pieces[:,1], marker='x', c='black', s = 20, label='pieces')
plt.scatter(centers[:,0], centers[:,1], marker='o', c='red', s = 15, label='cluster centers')
for i in range(pieces.shape[0]):
    ax.annotate(str(i+1), (pieces[i,0]-2, pieces[i,1]+0.2))
plt.legend(loc=1)
plt.text(centers[0,0], centers[0,1]+1.2, 'a', ha='center', fontweight='bold', wrap=True)
plt.text(centers[1,0], centers[1,1]+1.2, 'b', ha='center', fontweight='bold', wrap=True)
plt.text(centers[2,0], centers[2,1]+1.2, 'c', ha='center', fontweight='bold', wrap=True)
plt.tight_layout()
plt.savefig('digitization1.pdf', dpi=300, transparent=True)


# Plot digitization with scl=inf
abba = ABBA(tol=0.4, scl=np.inf, min_k=3, max_k=10, verbose=1)
string, centers = abba.transform(ts)
print(string)
fig, ax = myfigure(nrows=1, ncols=1, fig_ratio=0.8, fig_scale=1.7)
ax.axis([0, 80, -8, 6])
plt.xlabel('length')
plt.ylabel('increment')
split1 = (centers[0,0] + centers[1,0])/2
split2 = (centers[1,0] + centers[2,0])/2
ax.axvspan(0, split1, color=c2, zorder=1)
ax.axvspan(split1, split2, color=c1, zorder=1)
ax.axvspan(split2, 80, color=c3, zorder=1)
plt.scatter(pieces[:,0], pieces[:,1], marker='x', c='black', s = 20, label='pieces', zorder=2)
plt.scatter(centers[:,0], centers[:,1], marker='o', c='red', s = 15, label='cluster centers', zorder=2)
plt.text(centers[0,0], -4, 'a', ha='center', fontweight='bold', wrap=True)
plt.text(centers[1,0], -4, 'b', ha='center', fontweight='bold', wrap=True)
plt.text(centers[2,0], -4, 'c', ha='center', fontweight='bold', wrap=True)
for i in range(pieces.shape[0]):
    ax.annotate(str(i+1), (pieces[i,0]-2, pieces[i,1]+0.2))
plt.legend(loc=1)
plt.tight_layout()
plt.savefig('digitizationinf.pdf', dpi=300, transparent=True)
