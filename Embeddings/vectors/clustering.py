"""
Co-occurrence clustering and distance measures
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 June 24th"

import numpy as np
from scipy.sparse import csr_matrix
import scipy.io
import sklearn.decomposition
from sklearn.utils import gen_batches
from w2v import w2v
import gensim
import cPickle
import logging
import os
import gc
import metrics
import sys
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = '../data/'
MODEL_DIR = '../data/model/'
TEMP_DIR = '/phys/temp/'
N_ARTICLES = 4667756
N_DOCUMENTS = 137964149


class SparseMatrix(object):
    def __init__(self):
        self.row = None
        self.col = None
        self.data = None
        self.matrix = None
        self.ptr = 0

    def init(self, size=494341156):
        self.row = np.zeros(size, dtype=np.uint32)
        self.col = np.zeros(size, dtype=np.uint32)
        self.data = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def convert(self, shape=()):
        self.matrix = csr_matrix((self.data, (self.row, self.col)), shape=shape)
        del self.row
        del self.col
        del self.data

    def save(self, filename):
        print 'Saving...'
        np.savez(filename, self.matrix.data, self.matrix.indices, self.matrix.indptr, self.matrix.shape)

    def load(self, filename):
        print 'Loading...'
        loader = np.load(filename)
        self.matrix = csr_matrix((loader['arr_0'], loader['arr_1'], loader['arr_2']), shape=loader['arr_3'])
        del loader

    def put_on_disk(self, folder):
        """
        Can only be called after conver() or load() has been called!
        """
        data_d = np.memmap(folder + 'data_temp.dat', dtype='float32', mode='w+', shape=self.matrix.data.shape)
        indices_d = np.memmap(folder + 'row_temp.dat', dtype='int32', mode='w+', shape=self.matrix.indices.shape)
        indptr_d = np.memmap(folder + 'col_temp.dat', dtype='int32', mode='w+', shape=self.matrix.indptr.shape)

        data_d[:] = self.matrix.data[:]
        indices_d[:] = self.matrix.indices[:]
        indptr_d[:] = self.matrix.indptr[:]

        self.matrix.data = data_d
        self.matrix.indices = indices_d
        self.matrix.indptr = indptr_d

        gc.collect()

    def update(self, r, c, d):
        self.row[self.ptr] = r
        self.col[self.ptr] = c
        self.data[self.ptr] = d
        self.ptr += 1

    def get(self, i, j):
        return self.matrix.get((i, j), 0.0)



# DICT version -> less words, fits in memory
def update_matrix(S, dictionary, text, doc):
    l = text.split()
    lt = list()
    for word in l:
        index = dictionary.token2id.get(word, -1)
        if index >= 0:
            lt.append(index)
    v = dict()
    u = set(lt)
    c = get_frequency_table(lt)
    for x in u:
        v[(x, doc)] = c[x] * (np.log(N_DOCUMENTS) - np.log(dictionary.dfs[x]))
    S.update(v)

# CSR version
def update_matrix_csr(S, dictionary, text, doc):
    l = text.split()
    lt = list()
    for word in l:
        index = dictionary.token2id.get(word, -1)
        if index >= 0:
            lt.append(index)
    u = set(lt)
    c = get_frequency_table(lt)
    for x in u:
        S.update(x, doc, c[x] * (np.log(N_DOCUMENTS) - np.log(dictionary.dfs[x])))

def get_frequency_table(l):
    c = dict()
    for x in l:
        c[x] = c.get(x, 0) + 1
    return c

def fill_matrix(S, dictionary, file):
    processed = 0
    document = 0
    for line in open(file, 'r'):
        processed += 1
        if processed % 10000 == 0:
            logging.info('Processing document ' + str(processed))
        if processed % 2 == 0:
            continue
        update_matrix_csr(S, dictionary, line, document)
        document += 1

def count_matrix(dictionary, file):
    processed = 0
    counter = 0
    for line in open(file, 'r'):
        processed += 1
        if processed % 10000 == 0:
            logging.info('Processing document ' + str(processed) + ', ' + str(counter))
        if processed % 2 == 0:
            continue
        counter += count(dictionary, line)
    logging.info('Final counter ' + str(counter))
    return counter

def count(dictionary, text):
    l = text.split()
    lt = list()
    for word in l:
        index = dictionary.token2id.get(word, -1)
        if index >= 0:
            lt.append(index)
    u = set(lt)
    return len(u)

def do_pca(S, output_filename):
    model = sklearn.decomposition.IncrementalPCA(n_components=400, batch_size=300)
    model.fit(S.matrix)

    res = []
    for batch in gen_batches(S.matrix.shape[0], 400):
        res.append(S.matrix[batch].todense()[:, 0::4])
    output = np.vstack(res)

    f = open(output_filename, 'wb')
    np.savez(f, output)
    f.close()

if __name__ == '__main__':
    dictionary = gensim.corpora.Dictionary.load(MODEL_DIR + 'wiki_wordids_filtered_2.dict')
    dictionary.num_docs = metrics.N_DOCUMENTS

    S = SparseMatrix()
    S.init(size=494341156)

    fill_matrix(S, dictionary, TEMP_DIR + 'enwiki-articles.txt')

    S.convert((len(dictionary), N_ARTICLES))
    S.save(TEMP_DIR + 'cooc.npz')

    do_pca(S, 'pca.npz')