"""
Experiments 2 w2v
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 20th"


from multiprocessing import Process
import numpy as np
import scipy.spatial.distance
import math
import matplotlib
from w2v import w2v
import metrics
import similarity_plots as sp
import matplotlib.pyplot as plt




def make_plot_from_vector(vector=np.zeros(1), label=['word'], output='plot.pdf'):
    print 'Creating plot ' + output + ' ...'
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(0, vector.shape[0]-1, vector.shape[0])
    plt.plot(x, vector, 'bs', label=label[0])
    plt.axis([-1, vector.shape[0], -0.1, 1.1])

    #plt.legend()
    plt.savefig(output)
    plt.close()
    print 'Done.'

def make_plot_from_table(tables=['../data/tfidf-pairs.npy'], labels=['pairs'], colors=['red'], output='plot.pdf', log=False):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if log:
        ax.set_yscale('log', basey=10)
    _, _, _ = plt.hist(data, 300, normed=0, histtype='step', label=labels, color=colors)
    plt.axis([0.6, 1.0, 0, 300000])
    plt.xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    #ax.set_xlim(0.0, 1.0)
    plt.savefig(output)



f = open('../data/model/docfreq.npy')
docfreqs = np.load(f)
f.close()

w = w2v()
#w.load_minimal('../data/model/minimal')

texts = ['../data/pairs/enwiki_no_pairs_10.txt', '../data/pairs/enwiki_pairs_10.txt']
labels = ['Pairs', 'No pairs']
colors = ['0.75', '0.45']


input = ['../data/pairs/mean_no_pairs_20.npy', '../data/pairs/nntop_no_pairs_20.npy',\
        '../data/pairs/mean_pairs_20.npy', '../data/pairs/nntop_pairs_20.npy']
labels = ['No pairs', 'No pairs', 'Pairs', 'Pairs']
colors = ['0.60', '0.60', '0.10', '0.10']
make_plot_from_table(input, labels=labels, colors=colors, output='mean-nntop-20-comparison.pdf', log=False)