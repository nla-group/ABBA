"""
This demonstrates the use of the greedy ABBA 1d clustering approach.
"""

import sys
sys.path.append('./..')
import numpy as np
from numpy.linalg import norm
np.random.seed(0)
import matplotlib.pyplot as plt
%matplotlib inline
from ABBA import ABBA
from tslearn.metrics import dtw as dtw
from tslearn.metrics import dtw_path as dtw_path

# %%
# We're using L1 norms here, so the initial normalization is to have 0 median
# and unit mean deviation from zero.
ts = np.sin(7*np.linspace(0,4*1.8*np.pi,2400))
ts = np.array(ts)
ts -= np.median(ts)
ts *= len(ts)/norm(ts,1)

# %%
# We now compress ts into a polygon.
# Note the norm=1 parameter given to the constructor.
TOL = .333
abba = ABBA(tol=TOL, norm=1);
pieces = abba.compress(ts)
polyg = abba.inverse_compress(ts[0], pieces)
polyg = np.array(polyg)
print('one norm error of polygon:', norm(polyg-ts,1), ' -- should be bounded by:', TOL*(len(ts)+1-len(pieces)))

plt.figure()
plt.plot(ts, 'k-')
plt.plot(polyg, 'r--')
plt.title('original time series and polygon approximation')
plt.axis('tight');

# %%
# Now we run the digitize_inc procedure to cluster the pieces according
# their increment values.
string, centers = abba.digitize_inc(pieces)
print('Polygon now represented by the string', string)
approx = abba.inverse_transform(string, centers, ts[0])
approx = np.array(approx)

plt.figure()
plt.plot(polyg, 'r--', label='polygon')
plt.plot(approx, 'b-', label='reconstruction')
plt.title('polygon vs ABBA reconstruction')
plt.legend(); plt.axis('tight');

# %%
# Use average shape of each piece for the reconstruction and plot.
patches = abba.get_patches(ts, pieces, string, centers)
plt.figure()
abba.plot_patches(patches, string, centers, ts[0])

# %%
# verify increment deviation for cluster globally and per cluster
# overwrite last column of pieces (containing approx err) wth cluster increment
for j in range(len(string)):
    lab = ord(string[j])-97 # label
    pieces[j,2] = centers[lab,1]

inc1 = pieces[:,1]
inc2 = pieces[:,2]
err = norm(np.cumsum(inc1) - np.cumsum(inc2),1)
print('accumulated increment errors:', err, '--- hopeful bound:', TOL*len(pieces)*len(centers)/2,"\n")

plt.figure()
labs = np.array([ord(j)-97 for j in string])
for k in range(len(centers)):
    ind = np.where(labs==k)[0]
    errc = np.cumsum(pieces[ind,1]) - np.cumsum(pieces[ind,2]) # error in current cluster
    plt.plot(errc)
    err = np.linalg.norm(errc,1)/len(ind)
    print('cluster', k+1, ' -- #pieces:', len(ind), ' -- error:', err, ' -- bound:', TOL)
plt.title('local increment error path on each cluster');

# %%
from collections import defaultdict

def dtw1(x, y, dist=lambda a, b: abs(a-b), return_path=False, filter_redundant=False):

    x = np.array(x)
    y = np.array(y)

    if filter_redundant:
        if return_path:
            warning.warn('return path not supported when filter_redundant=True')
            return_path = False

        # remove points
        if len(x) > 2:
            xdiff = np.diff(x)
            x = x[np.hstack((True,np.abs(xdiff[1:] - xdiff[0:-1]) >= 1e-14, True))]
        if len(y) > 2:
            ydiff = np.diff(y)
            y = y[np.hstack((True,np.abs(ydiff[1:] - ydiff[0:-1]) >= 1e-14, True))]

    len_x, len_y = len(x), len(y)
    window = [(i+1, j+1) for i in range(len_x) for j in range(len_y)]
    D = defaultdict(lambda: (float('inf'),))

    if return_path:
        D[0, 0] = (0, 0, 0)
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                          (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])

        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((i-1, j-1))
            i, j = D[i, j][1], D[i, j][2]
        path.reverse()
        return (D[len_x, len_y][0], path)

    else:
        D[0, 0] = 0
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min(D[i-1, j]+dt, D[i, j-1]+dt, D[i-1, j-1]+dt)
        return D[len_x, len_y]

# %%
TOL_ = np.logspace(-4,0,11)
bnd = []
bnd1 = []
npc = []
ncl = []
errtwo = []
errdtw0 = []
errdtw1 = []
for j in range(len(TOL_)):
    TOL = TOL_[j]
    print(TOL)
    abba = ABBA(tol=TOL, norm=1)
    pieces = abba.compress(ts)
    polyg = abba.inverse_compress(ts[0], pieces)
    polyg = np.array(polyg)
    err = dtw1(ts, polyg, filter_redundant=False)
    errdtw0.append(err)

    string, centers = abba.digitize_inc(pieces, tol=TOL)
    approx = abba.inverse_transform(string, centers, ts[0])
    approx = np.array(approx)

    bnd.append( TOL*(len(ts)-len(pieces)+1) )
    bnd1.append( TOL*len(ts)*len(centers)/2 )
    npc.append( len(pieces) )
    ncl.append( len(centers) )
    err = dtw1(approx, polyg, filter_redundant=True)
    errdtw1.append( err )

plt.figure();
plt.loglog(TOL_,bnd,'k--', label="compression bound")
plt.loglog(TOL_,bnd1,'k:', label="heuristic bound for digitze_inc")
plt.loglog(TOL_,errdtw0,'b-o', label='error of compression')
plt.loglog(TOL_,errdtw1,'r-o', label='error of digitize_inc result')
plt.legend(loc=4)
plt.axis('tight');
