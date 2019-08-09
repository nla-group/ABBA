import numpy as np
from collections import defaultdict
import warnings

def dtw(x, y, *, dist=lambda a, b: (a-b)*(a-b), return_path=False, filter_redundant=False):
    """
    Compute dynamic time warping distance between two time series x and y.

    Parameters
    ----------
    x - list
        First time series.
    y - list
        Second time series.
    dist - lambda function
        Lambda function defining the distance between two points of the time series.
        By default we use (x-y)^2 to correspond to literature standard for
        dtw. Note final distance d should be square rooted.
    return_path - bool
        Option to return tuple (d, path) where path is a list of tuples outlining
        the route through time series taken to compute dtw distance.
    filter_redundant - bool
        Control filtering to remove `redundant` time series due to sampling
        resolution. For example, if x = [0, 1, 2, 3, 4] and y = [0, 4]. The dynamic
        time series distance is non-zero. If filter_redundant=True then we remove
        the middle 3 time points from x where gradient is constant.

    Returns
    -------
    d - numpy float
        Summation of the dist(x[i], y[i]) along the optimal path to minimise overall
        distance. Standard dynamic time warping distance given by default dist and
        d**(0.5).
    path - list
        Path taken through time series.
    """

    x = np.array(x)
    y = np.array(y)

    if filter_redundant:
        if return_path:
            warnings.warn('return path not supported when filter_redundant=True')
            return_path = False

        # remove points
        if len(x) > 2:
            xdiff = np.diff(x)
            x = x[np.hstack((True,(xdiff[1:] - xdiff[0:-1]) >= 1e-14, True))]
        if len(y) > 2:
            ydiff = np.diff(y)
            y = y[np.hstack((True,(ydiff[1:] - ydiff[0:-1]) >= 1e-14, True))]

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
