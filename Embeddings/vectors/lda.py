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
N_DOCUMENTS = 4667757

class plain_corpus(gensim.corpora.textcorpus.TextCorpus):
    def __init__(self, textfile):
        self.textfile = textfile
        super(plain_corpus, self).__init__()

    def get_texts(self):
        for line in open(self.textfile):
            yield line.split()

    def __len__(self):
        return N_DOCUMENTS

class bow_corpus(gensim.corpora.textcorpus.TextCorpus):
    def __init__(self, textfile):
        self.textfile = textfile
        super(bow_corpus, self).__init__()

    def get_texts(self):
        if self.dictionary is None:
            logging.log(1, 'No dictionary set! Call set_dictionary() first.')
            sys.exit()
        for line in open(self.textfile):
            #yield self.dictionary.doc2bow(line.split())
            yield line.split()

    def __len__(self):
        return N_DOCUMENTS

    def set_dictionary(self, dictionary):
        self.dictionary = dictionary

class lda():
    def __init__(self):
        self.model = None

    def train(self, corpus=None, id2word=None, num_topics=400, update_every=1, eval_every=30, passes=1, chunksize=12000):
        logging.log(0, 'Begin training LDA model...')
        self.model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                                     chunksize=chunksize, passes=passes,
                                                     update_every=update_every, eval_every=eval_every)

    def save(self, filename):
        if self.model is not None:
            logging.log(0, 'Saving LDA model...')
            self.model.save(filename)
            logging.log(0, 'Done saving LDA model')
        else:
            logging.log(0, 'No model present. Execute train() function first.')

    def load(self, filename):
        logging.log(0, 'Loading LDA model...')
        self.model = gensim.models.ldamodel.LdaModel.load(filename)
        logging.log(0, 'Done loading LDA model')


def create_dictionary(textfile):
    c = plain_corpus(textfile)
    dictionary = gensim.corpora.dictionary.Dictionary(prune_at=None)
    dictionary.add_documents(c.get_texts(), prune_at=None)
    dictionary.save_as_text(MODEL_DIR + 'wiki_wordids.txt.bz2')

def filter_dictionary(dictionary):
    dictionary.filter_extremes(no_below=20, no_above=0.3, keep_n=200000)

def convert_bow_corpus(corpus, dictionary, outputfile):
    gensim.corpora.MmCorpus.serialize(fname=outputfile, corpus=corpus, id2word=dictionary, progress_cnt=10000)

if __name__ == '__main__':
    dict = gensim.corpora.Dictionary.load(MODEL_DIR + 'wiki_wordids_filtered_2.dict')
    dict.num_docs = N_DOCUMENTS

    #c = bow_corpus(TEMP_DIR + 'enwiki-articles.txt')
    #c.set_dictionary(dict)

    #convert_bow_corpus(c, dict, TEMP_DIR + 'enwiki-articles.mm')
    #del c

    c = gensim.corpora.MmCorpus(TEMP_DIR + 'enwiki-articles.mm')
    l = lda()
    l.train(corpus=c, id2word=dict, num_topics=400, eval_every=30, update_every=1, passes=1, chunksize=12000)
    l.save(TEMP_DIR + 'lda.model')