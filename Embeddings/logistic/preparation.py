__author__ = 'cedricdeboom'


import numpy as np
from vectors.w2v import w2v
from vectors.metrics import cosine



def prepare(input_pairs='pairs.txt', input_no_pairs='no_pairs.txt', output_npy='output.npy', n_datapoints=50000, max_length=20, w2v_model='minimal'):
    w = w2v()
    w.load_minimal(w2v_model)

    pf = open(input_pairs, 'r')
    npf = open(input_no_pairs, 'r')

    data = np.zeros((n_datapoints*2, max_length**2))

    for i in xrange(0, n_datapoints*2, 2):
        if i % 10000 == 0:
            print "Iterating " + str(i) + "th datapoint..."

        pair = pf.next().split(';')
        no_pair = npf.next().split(';')

        pairA = [k for k in pair[0].split() if w.exists_word(k)]
        pairB = [k for k in pair[1].split() if w.exists_word(k)]
        no_pairA = [k for k in no_pair[0].split() if w.exists_word(k)]
        no_pairB = [k for k in no_pair[1].split() if w.exists_word(k)]

        feature_index = 0
        for a in xrange(len(pairA)):
            for b in xrange(len(pairB)):
                data[i, feature_index] = cosine(w.get_vector(pairA[a]), w.get_vector(pairB[b]))
                feature_index += 1

        feature_index = 0
        for a in xrange(len(no_pairA)):
            for b in xrange(len(no_pairB)):
                data[i+1, feature_index] = cosine(w.get_vector(no_pairA[a]), w.get_vector(no_pairB[b]))
                feature_index += 1

    print "Saving..."
    np.save(output_npy, data)

    print "Done."
    pf.close()
    npf.close()


if __name__ == "__main__":
    input_pairs = "../data/wiki/pairs/sets/enwiki_pairs_20-train.txt"
    input_no_pairs = "../data/wiki/pairs/sets/enwiki_no_pairs_20-train.txt"
    output_npy = "../data/wiki/pairs/sets/enwiki_logistic_20-train.npy"
    n_datapoints = 50000
    max_length = 20
    w2v_model = "../data/wiki/model/minimal"

    prepare(input_pairs, input_no_pairs, output_npy, n_datapoints, max_length, w2v_model)