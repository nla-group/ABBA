import numpy as np
from collections import defaultdict
import warnings
import matplotlib as mpl
import matplotlib.font_manager

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
        # remove points
        if len(x) > 2:
            xdiff = np.diff(x)
            x_keep = np.abs(xdiff[1:] - xdiff[0:-1]) >= 1e-14
            x = x[np.hstack((True, x_keep, True))]
        else:
            x_keep = []

        if len(y) > 2:
            ydiff = np.diff(y)
            y_keep = np.abs(ydiff[1:] - ydiff[0:-1]) >= 1e-14
            y = y[np.hstack((True, y_keep, True))]
        else:
            y_keep = []

    len_x, len_y = len(x), len(y)
    window = [(i+1, j+1) for i in range(len_x) for j in range(len_y)]
    D = defaultdict(lambda: (float('inf'),))

    if return_path:
        if filter_redundant:
            x_ind = np.arange(1, len(x_keep)+1)
            y_ind = np.arange(1, len(y_keep)+1)
            x_ind = np.hstack((0, x_ind[x_keep], len(x_keep)+1))
            y_ind = np.hstack((0, y_ind[y_keep], len(y_keep)+1))
        else:
            x_ind = np.arange(len(x))
            y_ind = np.arange(len(y))

        D[0, 0] = (0, 0, 0)
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                          (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])

        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((x_ind[i-1], y_ind[j-1]))
            i, j = D[i, j][1], D[i, j][2]
        path.reverse()
        return (D[len_x, len_y][0], path)

    else:
        D[0, 0] = 0
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min(D[i-1, j]+dt, D[i, j-1]+dt, D[i-1, j-1]+dt)
        return D[len_x, len_y]


def myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1): # pragma: no cover
    """
    Parameters
    ----------
    nrows - int
        Number of rows (subplots)

    ncols - int
        Number of columns (subplots)

    fig_ratio - float
        Ratio between height and width

    fig_scale - float
        Scaling which magnifies font size

    Returns
    -------
    fig - matplotlib figure handle

    ax -  tuple of matplotlib axis handles

    Example
    -------
    from util import myfigure
    fig, (ax1, ax2) = myfigure(nrows=2, ncols=1)
    """
    size = 7

    l = 13.2/2.54
    fig, ax = mpl.pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(l/fig_scale, l*fig_ratio/fig_scale), dpi=80*fig_scale, facecolor='w', edgecolor='k')
    mpl.pyplot.subplots_adjust(left=0.11*fig_scale, right=1-0.05*fig_scale, bottom=0.085*fig_scale/fig_ratio, top=1-0.05*fig_scale/fig_ratio)

    # Use tex and correct font

    mpl.rcParams['font.serif'] = ['computer modern roman']
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    mpl.rcParams['font.size'] = size

    # MATLAB default (see MATLAB Axes Properties documentation)
    mpl.rcParams['legend.fontsize'] = size

    # remove margine padding on axis
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0

    mpl.pyplot.tight_layout(pad=1.3) # padding

    # Save fig with transparent background
    mpl.rcParams['savefig.transparent'] = True

    # Make legend frame border black and face white
    mpl.rcParams['legend.edgecolor'] = 'k'
    mpl.rcParams['legend.facecolor'] = 'w'
    mpl.rcParams['legend.framealpha'] = 1

    # Change colorcycle to MATLABS
    c = mpl.cycler(color=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F'])

    if isinstance(ax, np.ndarray):
        ax = ax.ravel()
        for axi in ax:
            axi.set_prop_cycle(c) # color cycle
            axi.xaxis.label.set_size(1.1*size) # xaxis font size
            axi.yaxis.label.set_size(1.1*size) # yaxis font size
            axi.tick_params(axis='both', which='both', labelsize=size, direction='in') # fix ticks
    else:
        ax.set_prop_cycle(c) # color cycle
        ax.tick_params(axis='both', which='both', labelsize=size, direction='in') # fix ticks
        ax.xaxis.label.set_size(1.1*size) # xaxis font size
        ax.yaxis.label.set_size(1.1*size) # yaxis font size

    return fig, ax
