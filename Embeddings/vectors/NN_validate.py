__author__ = 'cedricdeboom'


import sys
from multiprocessing import Process
import numpy as np

import NN_train_8
import NN_trained_weights
import similarity_plots as sp
import metrics
from w2v import w2v


lower = int(sys.argv[1])
upper = int(sys.argv[2])
step = int(sys.argv[3])
print_file = sys.argv[4]

f = open('../data/wiki/model/docfreq.npy')
docfreqs = np.load(f)
f.close()

f = open('../data/tweets/model/idf_weights.npy')
idf_weights = np.load(f)
f.close()

w = w2v()
w.load_minimal('../data/wiki/model/minimal')

for i in xrange(lower, upper, step):
    #Train
    n = NN_train_8.NN8()
    NN_train_8.COST_FACTOR = float(i)
    NN_train_8.LEARNING_RATE = 0.01
    weights = n.run(w2v_model=w)
    NN_trained_weights.EUCL_WEIGHTS_V = weights

    #Validate
    texts1 = ['../data/tweets/pairs/sets/tweet-no-pairs-train.txt', '../data/tweets/pairs/sets/tweet-pairs-train.txt']
    output1 = ['../data/tweets/pairs/sets/nntoptweets_no_pairs-train.npy', '../data/tweets/pairs/sets/nntoptweets_pairs-train.npy']
    p1 = Process(target=sp.process_to_file_with_filter, args=(metrics.NNVarMean(metrics.euclidean, length='V'), texts1, output1, 20100, w, docfreqs))
    p1.start()

    texts2 = ['../data/tweets/pairs/sets/tweet-no-pairs-test.txt', '../data/tweets/pairs/sets/tweet-pairs-test.txt']
    output2 = ['../data/tweets/pairs/sets/nntoptweets_no_pairs-test.npy', '../data/tweets/pairs/sets/nntoptweets_pairs-test.npy']
    p2 = Process(target=sp.process_to_file_with_filter, args=(metrics.NNVarMean(metrics.euclidean, length='V'), texts2, output2, 20100, w, docfreqs))
    p2.start()

    p1.join()
    p2.join()

    #Test
    tablesA = ['../data/tweets/pairs/sets/nntoptweets_no_pairs-train.npy', '../data/tweets/pairs/sets/nntoptweets_pairs-train.npy']
    minA, maxA = sp.calculate_min_max_from_table(tablesA)
    splitA = sp.calculate_split_from_table(tablesA, verbose=False, normalize=(minA, maxA))
    ev = sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))
    tablesA = ['../data/tweets/pairs/sets/nntoptweets_no_pairs-test.npy', '../data/tweets/pairs/sets/nntoptweets_pairs-test.npy']
    et = sp.calculate_error_rate_from_table(tablesA, splitA, normalize=(minA, maxA))

    #Write
    f = open(print_file, 'aw')
    print str(i) + '\t' + str(ev) + '\t' + str(et)
    f.write(str(i) + '\t' + str(ev) + '\t' + str(et) + '\n')
    f.close()