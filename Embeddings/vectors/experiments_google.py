#!/bin/bash
"""
Experiments w2v google news
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 October 22nd"


from multiprocessing import Process
import numpy as np
import scipy.spatial.distance
import math
import sys
import gensim
import matplotlib
from w2v import w2v
import metrics
import similarity_plots as sp


f = open('../data/google/model/docfreq.npy')
docfreqs = np.load(f)
f.close()

f = open('../data/tweets/model/idf_weights.npy')
idf_weights = np.load(f)
f.close()

w = w2v()
#w.load_minimal('../data/google/model/minimal')

labels = ['No pairs', 'Pairs']
colors = ['0.75', '0.45']


tables = ['../data/google/pairs/sets/tfidf_no_pairs_r-validation.npy', '../data/google/pairs/sets/tfidf_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/tfidf_no_pairs_r-test.npy', '../data/google/pairs/sets/tfidf_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_JS_from_table(tables, normalize=(min, max), verbose=True)

tables = ['../data/google/pairs/sets/mean_no_pairs_r-validation.npy', '../data/google/pairs/sets/mean_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='mean-google-val.png', log=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/mean_no_pairs_r-test.npy', '../data/google/pairs/sets/mean_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_hellinger_distance_from_table(tables, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='mean-google.png', log=False, normalize=(min, max))
print ""

tables = ['../data/google/pairs/sets/max_no_pairs_r-validation.npy', '../data/google/pairs/sets/max_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='max-google-val.png', log=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/max_no_pairs_r-test.npy', '../data/google/pairs/sets/max_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_hellinger_distance_from_table(tables, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='max-google.png', log=False, normalize=(min, max))
print ""

tables = ['../data/google/pairs/sets/minmaxcon_no_pairs_r-validation.npy', '../data/google/pairs/sets/minmaxcon_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='minmaxcon-google-val.png', log=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/minmaxcon_no_pairs_r-test.npy', '../data/google/pairs/sets/minmaxcon_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_hellinger_distance_from_table(tables, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='minmaxcon-google.png', log=False, normalize=(min, max))
print ""

tables = ['../data/google/pairs/sets/meantop_no_pairs_r-validation.npy', '../data/google/pairs/sets/meantop_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='meantop-google-val.png', log=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/meantop_no_pairs_r-test.npy', '../data/google/pairs/sets/meantop_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_hellinger_distance_from_table(tables, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='meantop-google.png', log=False, normalize=(min, max))
print ""

tables = ['../data/google/pairs/sets/maxtop_no_pairs_r-validation.npy', '../data/google/pairs/sets/maxtop_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='maxtop-google-val.png', log=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/maxtop_no_pairs_r-test.npy', '../data/google/pairs/sets/maxtop_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_hellinger_distance_from_table(tables, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='maxtop-google.png', log=False, normalize=(min, max))
print ""

tables = ['../data/google/pairs/sets/minmaxcontop_no_pairs_r-validation.npy', '../data/google/pairs/sets/minmaxcontop_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='minmaxcontop-google-val.png', log=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/minmaxcontop_no_pairs_r-test.npy', '../data/google/pairs/sets/minmaxcontop_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_hellinger_distance_from_table(tables, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='minmaxcontop-google.png', log=False, normalize=(min, max))
print ""

tables = ['../data/google/pairs/sets/wmean_no_pairs_r-validation.npy', '../data/google/pairs/sets/wmean_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='wmean-google-val.png', log=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/wmean_no_pairs_r-test.npy', '../data/google/pairs/sets/wmean_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_hellinger_distance_from_table(tables, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='wmean-google.png', log=False, normalize=(min, max))
print ""

tables = ['../data/google/pairs/sets/nntop_no_pairs_r-validation.npy', '../data/google/pairs/sets/nntop_pairs_r-validation.npy']
min, max = sp.calculate_min_max_from_table(tables)
split = sp.calculate_split_from_table(tables, verbose=False, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='nntop-google-val.png', log=False, normalize=(min, max))
tables = ['../data/google/pairs/sets/nntop_no_pairs_r-test.npy', '../data/google/pairs/sets/nntop_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tables, split, normalize=(min, max))
sp.calculate_hellinger_distance_from_table(tables, normalize=(min, max))
sp.make_plot_from_table(tables, labels=labels, colors=colors, output='nntop-google.png', log=False, normalize=(min, max))
print ""

# PROCESSING
# texts1 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-validation.txt']
# output1 = ['../data/google/pairs/sets/mean_no_pairs_r-validation.npy', '../data/google/pairs/sets/mean_pairs_r-validation.npy']
# p1 = Process(target=sp.process_to_file_with_filter, args=(metrics.mean(metrics.euclidean), texts1, output1, 2000000, w, docfreqs))
# p1.start()
#
# texts2 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-test.txt']
# output2 = ['../data/google/pairs/sets/mean_no_pairs_r-test.npy', '../data/google/pairs/sets/mean_pairs_r-test.npy']
# p2 = Process(target=sp.process_to_file_with_filter, args=(metrics.mean(metrics.euclidean), texts2, output2, 1600000, w, docfreqs))
# p2.start()
#
# texts3 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-validation.txt']
# output3 = ['../data/google/pairs/sets/max_no_pairs_r-validation.npy', '../data/google/pairs/sets/max_pairs_r-validation.npy']
# p3 = Process(target=sp.process_to_file_with_filter, args=(metrics.max(metrics.euclidean), texts3, output3, 2000000, w, docfreqs))
# p3.start()
#
# texts4 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-test.txt']
# output4 = ['../data/google/pairs/sets/max_no_pairs_r-test.npy', '../data/google/pairs/sets/max_pairs_r-test.npy']
# p4 = Process(target=sp.process_to_file_with_filter, args=(metrics.max(metrics.euclidean), texts4, output4, 1600000, w, docfreqs))
# p4.start()
#
# p1.join()
# p2.join()
# p3.join()
# p4.join()
#
#
# texts1 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-validation.txt']
# output1 = ['../data/google/pairs/sets/minmaxcon_no_pairs_r-validation.npy', '../data/google/pairs/sets/minmaxcon_pairs_r-validation.npy']
# p1 = Process(target=sp.process_to_file_with_filter, args=(metrics.minMax(metrics.euclidean), texts1, output1, 2000000, w, docfreqs))
# p1.start()
#
# texts2 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-test.txt']
# output2 = ['../data/google/pairs/sets/minmaxcon_no_pairs_r-test.npy', '../data/google/pairs/sets/minmaxcon_pairs_r-test.npy']
# p2 = Process(target=sp.process_to_file_with_filter, args=(metrics.minMax(metrics.euclidean), texts2, output2, 1600000, w, docfreqs))
# p2.start()
#
# texts3 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-validation.txt']
# output3 = ['../data/google/pairs/sets/meantop_no_pairs_r-validation.npy', '../data/google/pairs/sets/meantop_pairs_r-validation.npy']
# p3 = Process(target=sp.process_to_file_with_filter, args=(metrics.meanTop(metrics.euclidean), texts3, output3, 2000000, w, docfreqs))
# p3.start()
#
# texts4 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-test.txt']
# output4 = ['../data/google/pairs/sets/meantop_no_pairs_r-test.npy', '../data/google/pairs/sets/meantop_pairs_r-test.npy']
# p4 = Process(target=sp.process_to_file_with_filter, args=(metrics.meanTop(metrics.euclidean), texts4, output4, 1600000, w, docfreqs))
# p4.start()
#
# p1.join()
# p2.join()
# p3.join()
# p4.join()
#
# texts1 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-validation.txt']
# output1 = ['../data/google/pairs/sets/maxtop_no_pairs_r-validation.npy', '../data/google/pairs/sets/maxtop_pairs_r-validation.npy']
# p1 = Process(target=sp.process_to_file_with_filter, args=(metrics.maxTop(metrics.euclidean), texts1, output1, 2000000, w, docfreqs))
# p1.start()
#
# texts2 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-test.txt']
# output2 = ['../data/google/pairs/sets/maxtop_no_pairs_r-test.npy', '../data/google/pairs/sets/maxtop_pairs_r-test.npy']
# p2 = Process(target=sp.process_to_file_with_filter, args=(metrics.maxTop(metrics.euclidean), texts2, output2, 1600000, w, docfreqs))
# p2.start()
#
# texts3 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-validation.txt']
# output3 = ['../data/google/pairs/sets/minmaxcontop_no_pairs_r-validation.npy', '../data/google/pairs/sets/minmaxcontop_pairs_r-validation.npy']
# p3 = Process(target=sp.process_to_file_with_filter, args=(metrics.minMaxTop(metrics.euclidean), texts3, output3, 2000000, w, docfreqs))
# p3.start()
#
# texts4 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-test.txt']
# output4 = ['../data/google/pairs/sets/minmaxcontop_no_pairs_r-test.npy', '../data/google/pairs/sets/minmaxcontop_pairs_r-test.npy']
# p4 = Process(target=sp.process_to_file_with_filter, args=(metrics.minMaxTop(metrics.euclidean), texts4, output4, 1600000, w, docfreqs))
# p4.start()
#
# p1.join()
# p2.join()
# p3.join()
# p4.join()
#
# texts1 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-validation.txt']
# output1 = ['../data/google/pairs/sets/tfidf_no_pairs_r-validation.npy', '../data/google/pairs/sets/tfidf_pairs_r-validation.npy']
# p1 = Process(target=sp.process_to_file_with_filter, args=(metrics.tfidf(metrics.euclidean), texts1, output1, 2000000, w, docfreqs))
# p1.start()
#
# texts2 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-test.txt']
# output2 = ['../data/google/pairs/sets/tfidf_no_pairs_r-test.npy', '../data/google/pairs/sets/tfidf_pairs_r-test.npy']
# p2 = Process(target=sp.process_to_file_with_filter, args=(metrics.tfidf(metrics.euclidean), texts2, output2, 1600000, w, docfreqs))
# p2.start()
#
# texts3 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-validation.txt']
# output3 = ['../data/google/pairs/sets/nntop_no_pairs_r-validation.npy', '../data/google/pairs/sets/nntop_pairs_r-validation.npy']
# p3 = Process(target=sp.process_to_file_with_filter, args=(metrics.NNMean(metrics.euclidean, 'R'), texts3, output3, 2000000, w, docfreqs))
# p3.start()
#
# texts4 = ['../data/wiki/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_r-test.txt']
# output4 = ['../data/google/pairs/sets/nntop_no_pairs_r-test.npy', '../data/google/pairs/sets/nntop_pairs_r-test.npy']
# p4 = Process(target=sp.process_to_file_with_filter, args=(metrics.NNMean(metrics.euclidean, 'R'), texts4, output4, 1600000, w, docfreqs))
# p4.start()
#
# p1.join()
# p2.join()
# p3.join()
# p4.join()