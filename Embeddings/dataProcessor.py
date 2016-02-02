"""
Data processor for tweets
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 February 11th"

import numpy as np
from random import randint
import re
import os
import cPickle

import theano


class dataProcessor():
    DATA = None
    TWEETDIM = 140
    CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',\
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',\
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',\
         '#', '@', '$', '&', '\'', ' ']
    NUMCHARS = len(CHARS)

    def __init__(self):
        pass

    @staticmethod
    def getData(input_file, output_file, number):
        #check if dump file already exists; if yes, return its contents
        if os.path.isfile(output_file):
            print "Loading dataset..."
            f = open(output_file, 'r')
            dictionary = np.load(f)
            dataset = dictionary['arr_0']
            dataset_shift = dictionary['arr_1']
            f.close()
            print "Done."
            dataProcessor.DATA = dataset
            return (dataset, dataset_shift)

        n_tweets = 0
        with open(input_file) as f:
            for line in f:
                #count number of tweets
                n_tweets += 1

        n_tweets = min(n_tweets, number)
        print "Number of tweets: %d" % (n_tweets)

        #allocate memory for tweets
        dataset = np.zeros([n_tweets, dataProcessor.TWEETDIM])
        dataset_shift = np.zeros([n_tweets, dataProcessor.TWEETDIM])

        tweet_index = 0
        with open(input_file) as f:
            for line in f:
                if tweet_index%100000 == 0:
                    print "Processing tweet %d" % (tweet_index + 1)
                #Example line: id="516577332946288641";cr="1411996608380";text="Wie o wie";us="365633279";lat="50.854306";lon="4.401551";repl="-1";retw="false";
                indexCR2 = line.index('\";text=\"')
                indexText2 = line.index('\";us=\"', indexCR2+1)
                text = line[indexCR2+8:indexText2].decode('utf8')
                text = text.lower()
                text = re.sub('[[a-z0-9]+&[a-z0-9]+\s','&', text)
                text = re.sub('[\s&[a-z0-9]+\s',' & ', text)

                char_index = (dataProcessor.TWEETDIM - len(text)) / 2.0 #put tweet text in middle of 140-length buffer
                rand_index = randint(0, int(char_index * 2))
                char_index = int(char_index)
                for c in text:
                    dataset[tweet_index, char_index] = dataProcessor.CHARS.index(c) + 1
                    dataset_shift[tweet_index, rand_index] = dataProcessor.CHARS.index(c) + 1
                    char_index += 1
                    rand_index += 1
                tweet_index += 1

                if tweet_index >= n_tweets:
                    break

        dataProcessor.DATA = dataset

        print "Saving dataset..."
        f = open(output_file, 'w')
        np.savez_compressed(f, dataset, dataset_shift)
        f.close()
        print "Done."

        return (dataset, dataset_shift)

    @staticmethod
    def getShiftedData(input_file, output_file, number):
        shifted = np.zeros(dataProcessor.DATA.shape)

        for l in xrange(len(dataProcessor.DATA)):
            shifted[l,:] = np.roll(dataProcessor.DATA[l,:], randint(0, dataProcessor.TWEETDIM-1))

        return shifted

    @staticmethod
    def transformBatch(batch):
        dims = batch.shape
        transformed_batch = np.zeros(batch.shape + (dataProcessor.NUMCHARS,), dtype=theano.config.floatX)
        for i in xrange(dims[0]):
            for j in xrange(dims[1]):
                if batch[i, j] == 0:
                    continue
                transformed_batch[i, j, batch[i, j]-1] = 1
        return transformed_batch

    @staticmethod
    def readSimTweets(dumpfile):
        f = open(dumpfile, 'r')
        tweets = cPickle.load(f)
        sims = cPickle.load(f)
        f.close()

        dataset = []
        for t in tweets:
            temp = np.zeros([2, dataProcessor.TWEETDIM])
            char_index = (dataProcessor.TWEETDIM - len(t[0])) / 2.0 #put tweet text in middle of 140-length buffer
            char_index = int(char_index)
            for c in t[0]:
                temp[0, char_index] = dataProcessor.CHARS.index(c) + 1
                char_index += 1
            char_index = (dataProcessor.TWEETDIM - len(t[1])) / 2.0 #put tweet text in middle of 140-length buffer
            char_index = int(char_index)
            for c in t[1]:
                temp[1, char_index] = dataProcessor.CHARS.index(c) + 1
                char_index += 1
            dataset.append(temp)

        return (dataset, sims)


if __name__ == '__main__':
    dataProcessor.getData('data/london.txt', 'data/london.npz', 1000)