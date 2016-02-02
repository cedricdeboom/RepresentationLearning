"""
LDA processing
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 14th"

import numpy as np
import gensim
import cPickle
import logging
import os
import metrics
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = '../data/'
MODEL_DIR = '../data/model/'
TEMP_DIR = '/phys/temp/'
N_DOCUMENTS = 4667756


class lsi():
    def __init__(self):
        self.model = None

    def train(self, corpus=None, id2word=None, num_topics=400, chunksize=20000):
        logging.log(0, 'Begin training LSI model...')
        self.model = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                                     chunksize=chunksize)

    def save(self, filename):
        if self.model is not None:
            logging.log(0, 'Saving LSI model...')
            self.model.save(filename)
            logging.log(0, 'Done saving LSI model')
        else:
            logging.log(0, 'No model present. Execute train() function first.')

    def load(self, filename):
        logging.log(0, 'Loading LSI model...')
        self.model = gensim.models.lsimodel.LsiModel.load(filename)
        logging.log(0, 'Done loading LSI model')

    def __getitem__(self, item):
        return self.model[item]


if __name__ == '__main__':
    dict = gensim.corpora.Dictionary.load(MODEL_DIR + 'wiki_wordids_filtered_2.dict')
    dict.num_docs = metrics.N_DOCUMENTS

    #c = bow_corpus(TEMP_DIR + 'enwiki-articles.txt')
    #c.set_dictionary(dict)

    #convert_bow_corpus(c, dict, TEMP_DIR + 'enwiki-articles.mm')
    #del c

    c = gensim.corpora.MmCorpus(TEMP_DIR + 'enwiki-articles.mm')
    l = lsi()
    l.train(corpus=c, id2word=dict, num_topics=400, chunksize=20000)
    l.save(TEMP_DIR + 'lsi.model')