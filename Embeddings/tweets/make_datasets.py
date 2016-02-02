"""
Train, test and validation set creator for tweet pairs and no-pairs
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 Aug 26th"

import sys
import random

def make(pairs_file='tweet-pairs.txt',
         no_pairs_file='tweet-no-pairs.txt',
         n_train=15000,
         n_test=13645,
         n_validation=20000):
    p_file = open(pairs_file, 'r')
    np_file = open(no_pairs_file, 'r')

    pairs_list = []
    no_pairs_list = []

    for line in p_file:
        pairs_list.append(line)
    for line in np_file:
        no_pairs_list.append(line)

    p_file.close()
    np_file.close()

    random.shuffle(pairs_list)
    random.shuffle(no_pairs_list)
    counter = 0

    print 'Creating train set...'
    train_p_file = open(pairs_file[0:-4] + '-train.txt', 'w')
    train_np_file = open(no_pairs_file[0:-4] + '-train.txt', 'w')
    for i in xrange(n_train):
        if i%100000 == 0:
            print i
        train_p_file.write(pairs_list[counter])
        train_np_file.write(no_pairs_list[counter])
        counter += 1
    train_p_file.close()
    train_np_file.close()

    print 'Creating test set...'
    test_p_file = open(pairs_file[0:-4] + '-test.txt', 'w')
    test_np_file = open(no_pairs_file[0:-4] + '-test.txt', 'w')
    for i in xrange(n_test):
        if i%100000 == 0:
            print i
        test_p_file.write(pairs_list[counter])
        test_np_file.write(no_pairs_list[counter])
        counter += 1
    test_p_file.close()
    test_np_file.close()

    print 'Creating validation set...'
    validation_p_file = open(pairs_file[0:-4] + '-validation.txt', 'w')
    validation_np_file = open(no_pairs_file[0:-4] + '-validation.txt', 'w')
    for i in xrange(n_validation):
        if i%100000 == 0:
            print i
        validation_p_file.write(pairs_list[counter])
        validation_np_file.write(no_pairs_list[counter])
        counter += 1
    validation_p_file.close()
    validation_np_file.close()


if __name__ == '__main__':
    pf = sys.argv[1]
    npf = sys.argv[2]
    ntr = int(sys.argv[3])
    nte = int(sys.argv[4])
    nva = int(sys.argv[5])

    make(pf, npf, ntr, nte, nva)