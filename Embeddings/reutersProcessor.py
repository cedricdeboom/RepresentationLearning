"""
Data processor for tweets
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 March 13th"

import numpy as np
from random import randint
from itertools import izip
import re
import os

import theano


class reutersProcessor():
    DATA1 = None
    DATA2 = None
    TWEETDIM = 140
    CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',\
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',\
         '0', '!', ' ', '"', '%', '$', '\'', '&', ')', '(', '+', '*', '-',\
         ',', '/', '.', ';', ':', '=', '?', '>', '@', '[', ']', '_', '~']
    NUMCHARS = len(CHARS)

    def __init__(self):
        pass

    @staticmethod
    def getData(input_file1, input_file2, output_file, number):
        #check if dump file already exists; if yes, return its contents
        if os.path.isfile(output_file):
            print "Loading dataset..."
            f = open(output_file, 'r')
            dictionary = np.load(f)
            dataset1 = dictionary['arr_0']
            dataset2 = dictionary['arr_1']
            f.close()
            print "Done."
            reutersProcessor.DATA1 = dataset1
            reutersProcessor.DATA2 = dataset2
            return (dataset1, dataset2)

        n_tweets = 0
        with open(input_file1) as f:
            for line in f:
                #count number of tweets
                n_tweets += 1

        n_tweets = min(n_tweets, number)
        print "Number of tweets: %d" % (n_tweets)

        #allocate memory for tweets
        dataset1 = np.zeros([n_tweets, reutersProcessor.TWEETDIM])
        dataset2 = np.zeros([n_tweets, reutersProcessor.TWEETDIM])

        tweet_index = 0
        with open(input_file1) as f1, open(input_file2) as f2:
            for line1, line2 in izip(f1, f2):
                if tweet_index%100000 == 0:
                    print "Processing tweet %d" % (tweet_index + 1)
                #Example line: news corp.'s fox posted 0 rating points and a 0 percent share for the week, nielsen said.
                line1 = line1[0:-1]
                line2 = line2[0:-1]
                char_index1 = (reutersProcessor.TWEETDIM - len(line1))
                rand_index1 = randint(0, char_index1)
                char_index2 = (reutersProcessor.TWEETDIM - len(line2))
                rand_index2 = randint(0, char_index2)
                for c1 in line1:
                    dataset1[tweet_index, rand_index1] = reutersProcessor.CHARS.index(c1) + 1
                    rand_index1 += 1
                for c2 in line2:
                    dataset2[tweet_index, rand_index2] = reutersProcessor.CHARS.index(c2) + 1
                    rand_index2 += 1
                tweet_index += 1

                if tweet_index >= n_tweets:
                    break

        reutersProcessor.DATA1 = dataset1
        reutersProcessor.DATA2 = dataset2

        print "Saving dataset..."
        f = open(output_file, 'w')
        np.savez_compressed(f, dataset1, dataset2)
        f.close()
        print "Done."

        return (dataset1, dataset2)

    @staticmethod
    def getShiftedData(input_file, output_file, number):
        shifted1 = np.zeros(reutersProcessor.DATA1.shape)
        shifted2 = np.zeros(reutersProcessor.DATA2.shape)

        for l in xrange(len(reutersProcessor.DATA1)):
            shifted1[l,:] = np.roll(reutersProcessor.DATA1[l,:], randint(0, reutersProcessor.TWEETDIM-1))

        for l in xrange(len(reutersProcessor.DATA2)):
            shifted2[l,:] = np.roll(reutersProcessor.DATA2[l,:], randint(0, reutersProcessor.TWEETDIM-1))

        return (shifted1, shifted2)

    @staticmethod
    def transformBatch(batch):
        dims = batch.shape
        transformed_batch = np.zeros(batch.shape + (reutersProcessor.NUMCHARS,), dtype=theano.config.floatX)
        for i in xrange(dims[0]):
            for j in xrange(dims[1]):
                if batch[i, j] == 0:
                    continue
                transformed_batch[i, j, batch[i, j]-1] = 1
        return transformed_batch


    @staticmethod
    def charTypes(file1, file2):
        f1 = open(file1, 'r')
        f2 = open(file2, 'r')
        d = set()

        for line in f1:
            for c in line:
                d.add(c)
        f1.close()

        for line in f2:
            for c in line:
                d.add(c)
        f2.close()
        print d

if __name__ == '__main__':
    reutersProcessor.getData('data/reuters/outputa_1_test.txt', 'data/reuters/outputa_2_test.txt', 'data/reuters/pairsa_test.npz', 240000)
