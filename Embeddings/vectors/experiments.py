#!/bin/bash
"""
Experiments w2v
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 20th"


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




f = open('../data/wiki/model/docfreq.npy')
docfreqs = np.load(f)
f.close()

#w = w2v()
#w.load_minimal('../data/model/minimal')

tablesA = ['../data/google/pairs/sets/tfidf_no_pairs_r-validation.npy', '../data/google/pairs/sets/tfidf_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/tfidf_no_pairs_r-test.npy', '../data/google/pairs/sets/tfidf_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""

tablesA = ['../data/google/pairs/sets/mean_no_pairs_r-validation.npy', '../data/google/pairs/sets/mean_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/mean_no_pairs_r-test.npy', '../data/google/pairs/sets/mean_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""

tablesA = ['../data/google/pairs/sets/max_no_pairs_r-validation.npy', '../data/google/pairs/sets/max_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/max_no_pairs_r-test.npy', '../data/google/pairs/sets/max_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""

tablesA = ['../data/google/pairs/sets/minmaxcon_no_pairs_r-validation.npy', '../data/google/pairs/sets/minmaxcon_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/minmaxcon_no_pairs_r-test.npy', '../data/google/pairs/sets/minmaxcon_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""

tablesA = ['../data/google/pairs/sets/meantop_no_pairs_r-validation.npy', '../data/google/pairs/sets/meantop_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/meantop_no_pairs_r-test.npy', '../data/google/pairs/sets/meantop_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""

tablesA = ['../data/google/pairs/sets/maxtop_no_pairs_r-validation.npy', '../data/google/pairs/sets/maxtop_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/maxtop_no_pairs_r-test.npy', '../data/google/pairs/sets/maxtop_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""

tablesA = ['../data/google/pairs/sets/minmaxcontop_no_pairs_r-validation.npy', '../data/google/pairs/sets/minmaxcontop_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/minmaxcontop_no_pairs_r-test.npy', '../data/google/pairs/sets/minmaxcontop_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""

tablesA = ['../data/google/pairs/sets/wmean_no_pairs_r-validation.npy', '../data/google/pairs/sets/wmean_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/wmean_no_pairs_r-test.npy', '../data/google/pairs/sets/wmean_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""

tablesA = ['../data/google/pairs/sets/nntop_no_pairs_r-validation.npy', '../data/google/pairs/sets/nntop_pairs_r-validation.npy']
minA, maxA = sp.calculate_min_max_from_table(tablesA)
splitA = sp.calculate_split_from_table(tablesA, verbose=True, normalize=(minA, maxA))
tablesA = ['../data/google/pairs/sets/nntop_no_pairs_r-test.npy', '../data/google/pairs/sets/nntop_pairs_r-test.npy']
sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
sp.calculate_JS_from_table(tablesA, normalize=(minA, maxA), verbose=True)
print ""



#sp.calculate_bootstrap_test_from_table(tablesB, tablesA, splitB, splitA, True, 5000, (minB, maxB), (minA, maxA))



# LDA-code
# dictionary = gensim.corpora.Dictionary.load('../data/model/wiki_wordids_filtered_2.dict')
# dictionary.num_docs = metrics.N_DOCUMENTS
#
# #pca_model = np.load('../data/model/pca.model.npz')['arr_0']
# import lda
# lda_model = lda.lda()
# lda_model.load('../data/model/lda.model')
#
# texts = ['../data/pairs/enwiki_no_pairs_10.txt', '../data/pairs/enwiki_pairs_10.txt']
# labels = ['Pairs', 'No pairs']
# colors = ['0.75', '0.45']
#
# texts1 = ['../data/pairs/sets/enwiki_no_pairs_r-test.txt', '../data/pairs/sets/enwiki_pairs_r-test.txt']
# output1 = ['../data/pairs/sets/enwiki_no_pairs_r-test.npy', '../data/pairs/sets/enwiki_pairs_r-test.npy']
# p1 = Process(target=sp.process_batch_to_file, args=(metrics.LDA(metrics.euclidean, dictionary, lda_model), texts1, output1, 1600000, w, docfreqs))
# p1.start()
# p1.join()
#
# texts2 = ['../data/pairs/sets/enwiki_no_pairs_r-validation.txt', '../data/pairs/sets/enwiki_pairs_r-validation.txt']
# output2 = ['../data/pairs/sets/enwiki_no_pairs_r-validation.npy', '../data/pairs/sets/enwiki_pairs_r-validation.npy']
# p2 = Process(target=sp.process_batch_to_file, args=(metrics.LDA(metrics.euclidean, dictionary, lda_model), texts2, output2, 2000000, w, docfreqs))
# p2.start()
# p2.join()