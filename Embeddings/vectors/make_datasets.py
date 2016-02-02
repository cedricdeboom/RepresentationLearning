"""
Train, test and validation set creator for wiki pairs and no-pairs
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 28th"

import sys


def make(pairs_file='pairs.txt',
         no_pairs_file='no-pairs.txt',
         n_train=1500000,
         n_test=1500000,
         n_validation=1900000):
    p_file = open(pairs_file, 'r')
    np_file = open(no_pairs_file, 'r')

    print 'Creating train set...'
    train_p_file = open(pairs_file[0:-4] + '-train.txt', 'w')
    train_np_file = open(no_pairs_file[0:-4] + '-train.txt', 'w')
    for i in xrange(n_train):
        if i%100000 == 0:
            print i
        train_p_file.write(p_file.next())
        train_np_file.write(np_file.next())
    train_p_file.close()
    train_np_file.close()

    print 'Creating test set...'
    test_p_file = open(pairs_file[0:-4] + '-test.txt', 'w')
    test_np_file = open(no_pairs_file[0:-4] + '-test.txt', 'w')
    for i in xrange(n_test):
        if i%100000 == 0:
            print i
        test_p_file.write(p_file.next())
        test_np_file.write(np_file.next())
    test_p_file.close()
    test_np_file.close()

    print 'Creating validation set...'
    validation_p_file = open(pairs_file[0:-4] + '-validation.txt', 'w')
    validation_np_file = open(no_pairs_file[0:-4] + '-validation.txt', 'w')
    for i in xrange(n_validation):
        if i%100000 == 0:
            print i
        validation_p_file.write(p_file.next())
        validation_np_file.write(np_file.next())
    validation_p_file.close()
    validation_np_file.close()

    p_file.close()
    np_file.close()


if __name__ == '__main__':
    pf = sys.argv[1]
    npf = sys.argv[2]
    ntr = int(sys.argv[3])
    nte = int(sys.argv[4])
    nva = int(sys.argv[5])

    make(pf, npf, ntr, nte, nva)