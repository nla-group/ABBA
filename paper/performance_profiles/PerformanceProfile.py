import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./../..')
from util import myfigure

def PerformanceProfile(A, th_max, file_name, alg_legend, n_intervals=100,
            markevery=1):
    """
    Modified from https://github.com/StevenElsworth/PerformanceProfiles
    """
    m, n = A.shape
    minA = np.min(A, 1)

    fig, ax = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1.6)

    # Use definition at some points theta
    n_intervals = 100;
    p_step = 1; # increase to make plot look smoother.
    theta = np.linspace(1, th_max, num = n_intervals);
    T = np.zeros([n_intervals,1]);

    linestyle = ['-', '--', '-.', ':']
    marker = ['o', '^', '*', 'x', 'v', '<', 'h', 'p']
    for j in np.arange(0,n):
        for k in np.arange(0,n_intervals):
            T[k] = np.sum(A[:,j] <= theta[k]*minA)/m;

        plt.plot(theta, T, linestyle=linestyle[j%4], marker=marker[j%8], markevery=markevery)

    plt.xlim([1, th_max])
    plt.ylim([0, 1.01])
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p$')
    plt.grid()
    if not alg_legend == None:
        plt.legend(alg_legend, loc=4)
    plt.tight_layout()
    plt.savefig(file_name, facecolor='w', edgecolor='w')
