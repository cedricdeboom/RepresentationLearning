"""
Util functions
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 February 25th"


import scipy.stats
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pearsonCorrelation(dataset1, dataset2):
    (r, p) = scipy.stats.pearsonr(dataset1, dataset2)
    return (r, p)

def plotArray(W, name, index):
    plt.clf()
    _, _, _ = plt.hist(W.flatten(), 100, normed=1, facecolor='green', alpha=0.75)
    plt.savefig(name + '_' + str(index) + '.png')