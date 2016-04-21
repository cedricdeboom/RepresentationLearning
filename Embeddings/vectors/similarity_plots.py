"""
Plotting of similarity between wiki/tweet pairs
based on w2v vocab
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 1st"


from multiprocessing import Process
import numpy as np
import scipy.spatial.distance
import math
import time
import matplotlib
from w2v import w2v
import metrics
import NN_trained_weights

import gensim

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='Times', style='normal')


def process_pairs(sim_function, pairs='../data/enwiki_pairs.txt', N=5100000, w2v_model=None, docfreqs=None):
    ret = np.zeros(N)
    i = 0

    with open(pairs) as f:
        for pair in f:
            parts = pair.split(';')
            ret[i] = sim_function(parts[0], parts[1], w2v_model, docfreqs)
            i += 1
            if i % 100000 == 1:
                print 'Processing ' + str(i) + 'st pair...'

    return ret[0:i]

def process_pairs_with_filter(sim_function, pairs='../data/enwiki_pairs.txt', N=5100000, w2v_model=None, docfreqs=None):
    ret = np.zeros(N)
    i = 0

    with open(pairs) as f:
        for pair in f:
            parts = pair.split(';')
            parts[0] = ' '.join([k for k in parts[0].split() if w2v_model.exists_word(k)]) #filter out words that do not appear in w2v model
            parts[1] = ' '.join([k for k in parts[1].split() if w2v_model.exists_word(k)]) #filter out words that do not appear in w2v model
            ret[i] = sim_function(parts[0], parts[1], w2v_model, docfreqs)
            i += 1
            if i % 100000 == 1:
                print 'Processing ' + str(i) + 'st pair...'

    return ret[0:i]

def process_batch_pairs(sim_function, pairs='../data/enwiki_pairs.txt', N=5100000, w2v_model=None, docfreqs=None):
    ret = np.zeros(N)
    temp = [None for k in xrange(10000 * 2)]
    t = 0
    i = 0

    with open(pairs) as f:
        for pair in f:
            parts = pair.split(';')
            temp[t*2], temp[t*2 + 1] = sim_function(parts[0], parts[1], w2v_model, docfreqs, batch=False)
            i += 1
            t += 1
            if i % 100000 == 1:
                print 'Processing ' + str(i) + 'st pair...'
            if t == 10000:
                s1 = time.time()
                ret[i-t : i] = sim_function(np.array(temp[0:t*2]), None, w2v_model, docfreqs, batch=True)
                print str(time.time() - s1) + ' ' + str(i)
                temp = [None for k in xrange(10000 * 2)]
                t = 0

    print 'Processing final batch...'
    ret[i-t : i] = sim_function(np.array(temp[0:t*2]), None, w2v_model, docfreqs, batch=True)
    print 'Done.'
    return ret[0:i]


def make_plot_from_text(sim_function, texts=['../data/enwiki_pairs.txt'], labels=['pairs'], colors=['red'], N=5100000, w2v_model=None, docfreqs=None, output='plot.png'):
    data = []
    for t in texts:
        r = process_pairs(sim_function, t, N, w2v_model, docfreqs)
        data.append(r)

    plt.clf()
    _, _, _ = plt.hist(data, 300, normed=1, histtype='step', label=labels, color=colors)
    #plt.legend()
    plt.savefig(output)

    print 'Calculating optimal error rate...'
    (error, split) = metrics.optimal_error_rate(data[0], data[1])
    print 'Optimal error: %.5f' % error
    print 'Optimal split point: %.5f' % split


def process_to_file(sim_function, texts=['../data/enwiki_pairs.txt'], output_files=['../data/tfidf-pairs.npy'], N=5100000, w2v_model=None, docfreqs=None):
    i = 0
    for t in texts:
        r = process_pairs(sim_function, t, N, w2v_model, docfreqs)
        f = open(output_files[i], 'wb')
        np.save(f, r)
        f.close()
        i += 1

def process_to_file_with_filter(sim_function, texts=['../data/enwiki_pairs.txt'], output_files=['../data/tfidf-pairs.npy'], N=5100000, w2v_model=None, docfreqs=None):
    i = 0
    for t in texts:
        r = process_pairs_with_filter(sim_function, t, N, w2v_model, docfreqs)
        f = open(output_files[i], 'wb')
        np.save(f, r)
        f.close()
        i += 1

def process_batch_to_file(sim_function, texts=['../data/enwiki_pairs.txt'], output_files=['../data/tfidf-pairs.npy'], N=5100000, w2v_model=None, docfreqs=None):
    i = 0
    for t in texts:
        r = process_batch_pairs(sim_function, t, N, w2v_model, docfreqs)
        f = open(output_files[i], 'wb')
        np.save(f, r)
        f.close()
        i += 1


def make_plot_from_table(tables=['../data/tfidf-pairs.npy'], labels=['pairs'], colors=['red'], output='plot.pdf', log=False, normalize=(0.0, 1.0)):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
        f.close()
    for d in xrange(len(data)):
        data[d] = (data[d] - normalize[0]) / (normalize[1] - normalize[0])

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if log:
        ax.set_yscale('log', basey=10)
    _, _, _ = plt.hist(data, 300, normed=0, histtype='step', label=labels, color=colors)
    ax.set_xlim(0.0, 1.0)
    #ax.set_ylim(0.0, 450.0)
    #leg = plt.legend(fontsize=16)
    #leg.get_frame().set_linewidth(0.0)
    plt.savefig(output)

    # print 'Calculating optimal error rate...'
    # (error, split) = metrics.optimal_error_rate(data[0], data[1])
    # print 'Optimal error: %.5f' % error
    # print 'Optimal split point: %.5f' % split

def make_hist_from_vector(vector=np.zeros(1), label=['word'], output='plot.png'):
    print 'Creating hist ' + output + ' ...'
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, _, _ = plt.hist(vector, 100, normed=1, histtype='bar', label=label)
    ax.set_xlim(-0.5, 0.5)
    plt.legend()
    plt.savefig(output)
    plt.close()
    print 'Done.'

def make_plot_from_vector(vectors=[np.zeros(1)], labels=['word'], colors=['bs'], output='plot.png'):
    print 'Creating plot ' + output + ' ...'
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for v in xrange(len(vectors)):
        x = np.linspace(0, vectors[v].shape[0]-1, vectors[v].shape[0])
        plt.plot(x, vectors[v], colors[v], label=labels[v])

    leg = plt.legend(fontsize=16)
    leg.get_frame().set_linewidth(0.0)
    plt.savefig(output)
    plt.close()
    print 'Done.'

def calculate_split_from_table(tables=['../data/tfidf-pairs.txt'], verbose=True, normalize=(0.0, 1.0)):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
    for d in xrange(len(data)):
        data[d] = (data[d] - normalize[0]) / (normalize[1] - normalize[0])

    if verbose:
        print 'Calculating optimal error rate...'
    (error, split) = metrics.optimal_error_rate(data[0], data[1])
    if verbose:
        print 'Optimal error: %.5f' % error
        print 'Optimal split point: %.5f' % split
    return split

def calculate_error_rate_from_table(tables=['../data/tfidf-pairs.npy'], split=0.0, verbose=True, normalize=(0.0, 1.0)):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
    for d in xrange(len(data)):
        data[d] = (data[d] - normalize[0]) / (normalize[1] - normalize[0])

    if verbose:
        print 'Calculating optimal error rate...'
    error = metrics.error_rate(data[0], data[1], split)
    if verbose:
        print 'Error: %.5f' % error
        print 'Used split point: %.5f' % split
    return error

def calculate_error_rate_variance_from_table(tables=['../data/tfidf-pairs.npy'], split=0.0, deviation=0.025, verbose=True, normalize=(0.0, 1.0)):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
    for d in xrange(len(data)):
        data[d] = (data[d] - normalize[0]) / (normalize[1] - normalize[0])

    if verbose:
        print 'Calculating error rate variance with a deviation of %.5f...' % deviation
    splits = np.linspace(split - deviation, split + deviation, 100)
    errors = []
    for s in splits:
        errors.append(metrics.error_rate(data[0], data[1], s))
    errors = np.asarray(errors)
    v = np.std(errors)
    m = np.max(errors)
    if verbose:
        print 'Stdev: %.5f' % v
        print 'Max value: %.5f' % m
        print 'Used split point: %.5f' % split
    return v, m

def calculate_prf_from_table(tables=['../data/tfidf-pairs.npy'], split=0.0, verbose=True, normalize=(0.0, 1.0)):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
    for d in xrange(len(data)):
        data[d] = (data[d] - normalize[0]) / (normalize[1] - normalize[0])

    if verbose:
        print 'Calculating P, R and F1...'
    p, r, f = metrics.prf(data[0], data[1], split)
    if verbose:
        print 'Precision: %.5f' % p
        print 'Recall: %.5f' % r
        print 'F1: %.5f' % f
        print 'Used split point: %.5f' % split
    return (p, r, f)

def calculate_binom_test_from_table(tablesA=['../data/tfidf-pairs.npy'], tablesB=['../data/tfidf-pairs.npy'], splitA=0.0, splitB=0.0, verbose=True, normalizeA=(0.0, 1.0),normalizeB=(0.0, 1.0)):
    dataA = []
    dataB = []
    for t in tablesA:
        f = open(t, 'rb')
        r = np.load(f)
        dataA.append(r)
    for d in xrange(len(dataA)):
        dataA[d] = (dataA[d] - normalizeA[0]) / (normalizeA[1] - normalizeA[0])
    for t in tablesB:
        f = open(t, 'rb')
        r = np.load(f)
        dataB.append(r)
    for d in xrange(len(dataB)):
        dataB[d] = (dataB[d] - normalizeB[0]) / (normalizeB[1] - normalizeB[0])
    if verbose:
        print 'Calculating statistics...'
    p = metrics.binom_test(dataA[0], dataA[1], dataB[0], dataB[1], splitA, splitB)
    if verbose:
        print 'p: %.10f' % p
        print 'Used split points: %.5f, %.5f' % (splitA, splitB)
    return p

def calculate_bootstrap_test_from_table(tablesA=['../data/tfidf-pairs.npy'], tablesB=['../data/tfidf-pairs.npy'], splitA=0.0, splitB=0.0, verbose=True, repeat=10000, normalizeA=(0.0, 1.0),normalizeB=(0.0, 1.0)):
    dataA = []
    dataB = []
    for t in tablesA:
        f = open(t, 'rb')
        r = np.load(f)
        dataA.append(r)
    for d in xrange(len(dataA)):
        dataA[d] = (dataA[d] - normalizeA[0]) / (normalizeA[1] - normalizeA[0])
    for t in tablesB:
        f = open(t, 'rb')
        r = np.load(f)
        dataB.append(r)
    for d in xrange(len(dataB)):
        dataB[d] = (dataB[d] - normalizeB[0]) / (normalizeB[1] - normalizeB[0])
    if verbose:
        print 'Calculating statistics...'
    p = metrics.bootstrap_test(dataA[0], dataA[1], dataB[0], dataB[1], splitA, splitB, repeat=repeat)
    if verbose:
        print 'p: %.10f' % p
        print 'Used split points: %.5f, %.5f' % (splitA, splitB)
    return p

def calculate_KL_from_table(tables=['../data/tfidf-pairs.npy'], verbose=True, normalize=(0.0, 1.0)):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
    for d in xrange(len(data)):
        data[d] = (data[d] - normalize[0]) / (normalize[1] - normalize[0])

    if verbose:
        print 'Calculating KL divergence...'
    k = metrics.KL_divergence(data[0], data[1])
    if verbose:
        print 'KL divergence: %.5f' % k
    return k

def calculate_JS_from_table(tables=['../data/tfidf-pairs.npy'], verbose=True, normalize=(0.0, 1.0)):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
    for d in xrange(len(data)):
        data[d] = (data[d] - normalize[0]) / (normalize[1] - normalize[0])

    if verbose:
        print 'Calculating JS divergence...'
    k = metrics.JS_divergence(data[0], data[1])
    if verbose:
        print 'JS divergence: %.5f' % k
    return k

def calculate_hellinger_distance_from_table(tables=['../data/tfidf-pairs.npy'], verbose=True, normalize=(0.0, 1.0)):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
    for d in xrange(len(data)):
        data[d] = (data[d] - normalize[0]) / (normalize[1] - normalize[0])

    if verbose:
        print 'Calculating hellinger distance...'
    k = metrics.hellinger_distance(data[0], data[1])
    if verbose:
        print 'Hellinger distance: %.5f' % k
    return k

def find_max(tables=['../data/tfidf-pairs.npy']):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)

    max = 0.0
    for d in data:
        t = np.max(d)
        if t > max:
            max = t
    return max

def get_mean_idf_vector_fixed_length(texts=['../data/enwiki_pairs.txt'], length=20, w2v_model=None, docfreqs=None):
    res = np.zeros(length+1)
    number_of_entries = 0.0

    for t in texts:
        with open(t, 'r') as f:
            for line in f:
                a, b = line.split(';')
                fA = get_sorted_idf_vector(a, w2v_model, docfreqs)
                fB = get_sorted_idf_vector(b, w2v_model, docfreqs)
                res += fA
                res += fB
                number_of_entries += 2.0

    res /= number_of_entries
    res /= np.max(res)

    return res

def get_mean_idf_vector_var_length(texts=['../data/enwiki_pairs.txt'], max_length=20, printout=False, w2v_model=None, docfreqs=None):
    res = np.zeros((max_length + 1, max_length + 1))
    number_of_entries = np.zeros(max_length + 1)

    for t in texts:
        with open(t, 'r') as f:
            for line in f:
                a, b = line.split(';')
                fA = get_sorted_idf_vector(a, w2v_model, docfreqs)
                fB = get_sorted_idf_vector(b, w2v_model, docfreqs)
                res[len(fA), 0:len(fA)] += fA
                res[len(fB), 0:len(fB)] += fB
                number_of_entries[len(fA)] += 1.0
                number_of_entries[len(fB)] += 1.0

    for i in xrange(max_length + 1):
        res[i] /= number_of_entries[i]
        res[i] /= np.max(res[i])

    if printout:
        for i in xrange(1, max_length + 1):
            for j in xrange(max_length):
                if res[i, j] > 1E-50:
                    print j
        print "\n\n"

        for i in xrange(1, max_length + 1):
            for j in xrange(max_length):
                if res[i, j] > 1E-50:
                    print res[i, j]
        print "\n\n"

        for i in xrange(1, max_length):
            for j in xrange(max_length):
                if res[i, j] > 1E-50:
                    print i

    return res

def get_tweet_weights_var_length(length=10, type='V', distance=metrics.euclidean, cutoff=None):
    WEIGHTS = np.asarray(NN_trained_weights.get_weights_variable_name(distance, type))
    if cutoff is None:
        a = np.linspace(0.0, 34.0, length)
    else:
        a = np.linspace(0.0, (1 - cutoff(length))*34.0, length)

    a_high = np.ceil(a).astype(np.int8)
    a_low = np.floor(a).astype(np.int8)
    af_high = WEIGHTS[a_high]
    af_low = WEIGHTS[a_low]
    af = ((af_high - af_low) * (a - a_low) / (a_high - a_low + 1E-5)) + af_low

    return af


def get_sorted_idf_vector(text='a b c', w2v_model=None, docfreqs=None):
    wordsA = text.split()
    wordsA = [k for k in wordsA if w2v_model.exists_word(k)]
    fA = [0]*len(wordsA)

    for i in xrange(len(wordsA)):
        fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]

    fA, _ = zip(*sorted(zip(fA, wordsA)))
    return (np.log(metrics.N_DOCUMENTS) - np.log(fA))

def print_random_pairs(texts=['../data/enwiki_pairs.txt'], number=100):
    for t in texts:
        length = 0
        with open(t, 'r') as f:
            for line in f:
                length += 1
        p = np.sort(np.random.randint(0, length, number))

        length = 0
        index = 0
        with open(t, 'r') as f:
            for line in f:
                if length == p[index]:
                    print line[0:-1]
                    index += 1
                if index == number:
                    break
                length += 1

def calculate_min_max_from_table(tables=['../data/tfidf-pairs.npy']):
    data = []
    for t in tables:
        f = open(t, 'rb')
        r = np.load(f)
        data.append(r)
        f.close()
    max = -float('inf')
    min = float('inf')
    for d in data:
        if np.max(d) > max:
            max = np.max(d)
        if np.min(d) < min:
            min = np.min(d)
    return min, max


if __name__ == '__main__':
    f = open('../data/wiki/model/docfreq.npy')
    docfreqs = np.load(f)
    f.close()

    f = open('../data/tweets/model/idf_weights.npy')
    idf_weights = np.load(f)
    f.close()

    w = w2v()
    w.load_minimal('../data/wiki/model/minimal')

    texts = ['../data/wiki/pairs/enwiki_no_pairs_10.txt', '../data/wiki/pairs/enwiki_pairs_10.txt']
    labels = ['Pairs', 'No pairs']
    colors = ['0.75', '0.45']

    # s = 'anarchism is a political philosophy that advocated stateless societies often self governed voluntary institutions but that several authors have defined'.split()
    # sv = []
    # for word in s:
    #     sv.append(w.get_vector(word))
    # for k in xrange(len(sv)):
    #     make_plot_from_vector(sv[k], s[k], 'vector_'+str(k)+'.png')

    # texts1 = ['../data/tweets/pairs/sets/tweet-no-pairs-train.txt', '../data/tweets/pairs/sets/tweet-pairs-train.txt']
    # r = get_mean_idf_vector_var_length(texts=texts1, max_length=35, printout=False, w2v_model=w, docfreqs=docfreqs)
    # f = open('../data/tweets/model/idf_weights.npy', 'wb')
    # np.save(f, r)
    # f.close()
    # np.set_printoptions(threshold=np.nan)
    # print r
    #
    # for i in xrange(len(r)):
    #     print r[i, 0] - r[i, i-1]


    # make_plot_from_vector(np.asarray(NN_trained_weights.EUCL_WEIGHTS_10), ['10'], '10.pdf')
    # make_plot_from_vector(np.asarray(NN_trained_weights.EUCL_WEIGHTS_20), ['20'], '20.pdf')
    # make_plot_from_vector(np.asarray(NN_trained_weights.EUCL_WEIGHTS_30), ['30'], '30.pdf')

    # make_plot_from_vector([np.asarray(NN_trained_weights.EUCL_WEIGHTS_20), np.asarray(NN_trained_weights.EUCL_WEIGHTS_20M)],
    #                       ['Contrastive loss', 'Median loss'], ['bs-', 'r^-'], 'test.pdf')

    texts1 = ['../data/tweets/pairs/sets/tweet-no-pairs-test.txt', '../data/tweets/pairs/sets/tweet-pairs-test.txt']
    output1 = ['../data/tweets/pairs/sets/nntopcontr_no_pairs-test.npy', '../data/tweets/pairs/sets/nntopcontr_pairs-test.npy']
    p1 = Process(target=process_to_file_with_filter, args=(metrics.NNVarMean(metrics.euclidean, 'VC'), texts1, output1, 100000, w, docfreqs))
    p1.start()

    texts2 = ['../data/tweets/pairs/sets/tweet-no-pairs-validation.txt', '../data/tweets/pairs/sets/tweet-pairs-validation.txt']
    output2 = ['../data/tweets/pairs/sets/nntopcontr_no_pairs-validation.npy', '../data/tweets/pairs/sets/nntopcontr_pairs-validation.npy']
    p2 = Process(target=process_to_file_with_filter, args=(metrics.NNVarMean(metrics.euclidean, 'VC'), texts2, output2, 100000, w, docfreqs))
    p2.start()
    #
    # texts3 = ['../data/wiki/pairs/sets/enwiki_no_pairs_20-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_20-test.txt']
    # output3 = ['../data/wiki/pairs/sets/minmaxcontop_no_pairs_20-test.npy', '../data/wiki/pairs/sets/minmaxcontop_pairs_20-test.npy']
    # p3 = Process(target=process_to_file_with_filter, args=(metrics.minMaxTop(metrics.euclidean), texts3, output3, 2000000, w, docfreqs))
    # p3.start()
    #
    # texts4 = ['../data/wiki/pairs/sets/enwiki_no_pairs_20-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_20-validation.txt']
    # output4 = ['../data/wiki/pairs/sets/minmaxcontop_no_pairs_20-validation.npy', '../data/wiki/pairs/sets/minmaxcontop_pairs_20-validation.npy']
    # p4 = Process(target=process_to_file_with_filter, args=(metrics.minMaxTop(metrics.euclidean), texts4, output4, 2000000, w, docfreqs))
    # p4.start()
    #
    p1.join()
    p2.join()
    # p3.join()
    # p4.join()
    #
    # texts1 = ['../data/wiki/pairs/sets/enwiki_no_pairs_20-test.txt', '../data/wiki/pairs/sets/enwiki_pairs_20-test.txt']
    # output1 = ['../data/wiki/pairs/sets/maxtop_no_pairs_20-test.npy', '../data/wiki/pairs/sets/maxtop_pairs_20-test.npy']
    # p1 = Process(target=process_to_file_with_filter, args=(metrics.maxTop(metrics.euclidean), texts1, output1, 2000000, w, docfreqs))
    # p1.start()
    #
    # texts2 = ['../data/wiki/pairs/sets/enwiki_no_pairs_20-validation.txt', '../data/wiki/pairs/sets/enwiki_pairs_20-validation.txt']
    # output2 = ['../data/wiki/pairs/sets/maxtop_no_pairs_20-validation.npy', '../data/wiki/pairs/sets/maxtop_pairs_20-validation.npy']
    # p2 = Process(target=process_to_file_with_filter, args=(metrics.maxTop(metrics.euclidean), texts2, output2, 2000000, w, docfreqs))
    # p2.start()
    #
    # p1.join()
    # p2.join()

    #calculate_error_rate_from_table(tables=['../data/pairs/mean_no_pairs_30.npy', '../data/pairs/mean_pairs_30.npy'])
    #calculate_error_rate_from_table(tables=['../data/pairs/max_no_pairs_30.npy', '../data/pairs/max_pairs_30.npy'])
    #calculate_error_rate_from_table(tables=['../data/pairs/concat_no_pairs_30.npy', '../data/pairs/concat_pairs_30.npy'])
    #calculate_error_rate_from_table(tables=['../data/pairs/tfidf_no_pairs_10.npy', '../data/pairs/tfidf_pairs_10.npy'])
    #calculate_error_rate_from_table(tables=['../data/pairs/tfidf_no_pairs_20.npy', '../data/pairs/tfidf_pairs_20.npy'])
    #calculate_error_rate_from_table(tables=['../data/pairs/tfidf_no_pairs_30.npy', '../data/pairs/tfidf_pairs_30.npy'])

    # COMPARE TWO METHODS
    # input = ['../data/tweets/pairs/sets/mean_no_pairs-validation.npy',
    #          '../data/tweets/pairs/sets/mean_pairs-validation.npy']
    # labels = ['No pairs', 'Pairs']
    # colors = ['0.45', '0.75']
    # min, max = calculate_min_max_from_table(input)
    # make_plot_from_table(input, labels=labels, colors=colors, output='tweet-mean.pdf', log=False, normalize=(min, max))

    #LDA EXAMPLE
    #input = ['../data/pairs/lda_no_pairs_10.npy', '../data/pairs/lda_pairs_10.npy']
    #make_plot_from_table(input, labels=labels, colors=colors, output='lda-10.png', log=False)
    #calculate_split_from_table(input)
    #calculate_JS_from_table(input)
    #calculate_error_rate_from_table(input, 0.032)

    # input = ['../data/pairs/sets/mean_no_pairs_r-validation.npy', '../data/pairs/sets/mean_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.698)
    #
    # input = ['../data/pairs/sets/max_no_pairs_r-validation.npy', '../data/pairs/sets/max_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.355)
    #
    # input = ['../data/pairs/sets/minmaxcon_no_pairs_r-validation.npy', '../data/pairs/sets/minmaxcon_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.699)
    #
    # input = ['../data/pairs/sets/meantop_no_pairs_r-validation.npy', '../data/pairs/sets/meantop_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.819)
    #
    # input = ['../data/pairs/sets/maxtop_no_pairs_r-validation.npy', '../data/pairs/sets/maxtop_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.587)
    #
    # input = ['../data/pairs/sets/minmaxcontop_no_pairs_r-validation.npy', '../data/pairs/sets/minmaxcontop_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.805)
    #
    # input = ['../data/pairs/sets/wmean_no_pairs_r-validation.npy', '../data/pairs/sets/wmean_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.897)
    #
    # input = ['../data/pairs/sets/nntop_no_pairs_r-validation.npy', '../data/pairs/sets/nntop_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.840)
    #
    # input = ['../data/pairs/sets/pca_no_pairs_r-validation.npy', '../data/pairs/sets/pca_pairs_r-validation.npy']
    # calculate_error_rate_from_table(input, 0.97559)