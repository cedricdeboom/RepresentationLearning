"""
NN processor for word embeddings
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 23rd"

import theano
import numpy as np
import abc
import metrics
from w2v import w2v
from threading import Condition


class abstractProcessor(object):
    def __init__(self):
        self.lock = Condition()
        self.cont = False
        self.ready = False
        self.stop = False

    def new_epoch(self):
        self.begin_of_new_epoch()
        self.lock.acquire()
        self.cont = True
        self.ready = False
        self.stop = False
        self.lock.notifyAll()
        self.lock.release()

    def process(self):
        while True:
            self.lock.acquire()
            while not self.cont:
                self.lock.wait()
            self.ready = False
            self.cont = False
            self.lock.release()
            if self.stop:
                break

            self.process_batch()

            self.lock.acquire()
            self.ready = True
            self.cont = False
            self.lock.notifyAll()
            self.lock.release()

    @abc.abstractmethod
    def begin_of_new_epoch(self):
        """Abstract"""
        raise NotImplementedError("Method begin_of_new_epoch is not implemented in this class.")

    @abc.abstractmethod
    def process_batch(self):
        """Abstract"""
        raise NotImplementedError("Method process_all_batches is not implemented in this class.")


class PairProcessor(abstractProcessor):
    def __init__(self, pairs_filename='pairs.txt', no_pairs_filename='no_pairs.txt', docfreq_filename='docfreqs.npy',
                 w2v_filename='minimal', no_words=20, embedding_dim=400, batch_size=100):

        super(PairProcessor, self).__init__()

        self.pairs_filename = pairs_filename
        self.no_pairs_filename = no_pairs_filename
        self.batch_size = batch_size
        self.no_words = no_words
        self.embedding_dim = embedding_dim
        self.docfreq_filename = docfreq_filename
        self.w2v_filename = w2v_filename

        self.x1 = np.zeros((batch_size, embedding_dim, no_words), dtype=theano.config.floatX)
        self.x2 = np.zeros((batch_size, embedding_dim, no_words), dtype=theano.config.floatX)
        self.y = np.zeros((batch_size), dtype=theano.config.floatX)
        self.z = np.zeros((batch_size), dtype=theano.config.floatX)

        f = open(self.docfreq_filename)
        self.docfreqs = np.load(f)
        f.close()

        if isinstance(w2v_filename, basestring):
            self.w = w2v()
            self.w.load_minimal(self.w2v_filename)
        else:
            self.w = w2v_filename

    def begin_of_new_epoch(self):
        try:
            self.pairs_file.close()
            if self.no_pairs_filename is not None:
                self.no_pairs_file.close()
        except:
            pass
        self.pairs_file = open(self.pairs_filename, 'r')
        if self.no_pairs_filename is not None:
            self.no_pairs_file = open(self.no_pairs_filename, 'r')

    def process_batch(self):
        for i in xrange(0, self.batch_size, 2):
            pair = self.pairs_file.next().split(';')
            no_pair = self.no_pairs_file.next().split(';')

            pairA = pair[0].split()
            pairB = pair[1].split()
            no_pairA = no_pair[0].split()
            no_pairB = no_pair[1].split()

            dA = [0]*self.no_words
            dB = [0]*self.no_words
            nA = [0]*self.no_words
            nB = [0]*self.no_words

            for k in xrange(self.no_words):
                dA[k] = self.docfreqs[self.w.model.vocab[pairA[k]].index]
                dB[k] = self.docfreqs[self.w.model.vocab[pairB[k]].index]
                nA[k] = self.docfreqs[self.w.model.vocab[no_pairA[k]].index]
                nB[k] = self.docfreqs[self.w.model.vocab[no_pairB[k]].index]

            _, pairA = zip(*sorted(zip(dA, pairA)))
            _, pairB = zip(*sorted(zip(dB, pairB)))
            _, no_pairA = zip(*sorted(zip(nA, no_pairA)))
            _, no_pairB = zip(*sorted(zip(nB, no_pairB)))

            for j in xrange(self.no_words):
                self.x1[i, :, j] = self.w.get_vector(pairA[j])
                self.x2[i, :, j] = self.w.get_vector(pairB[j])
                self.x1[i+1, :, j] = self.w.get_vector(no_pairA[j])
                self.x2[i+1, :, j] = self.w.get_vector(no_pairB[j])

            self.y[i] = 0.0
            self.z[i] = -1.0
            self.y[i+1] = 1.0
            self.z[i+1] = 1.0

class PairBisProcessor(PairProcessor):
    def __init__(self, pairs_filename='pairs.txt', no_pairs_filename='no_pairs.txt', docfreq_filename='docfreqs.npy',
                 w2v_filename='minimal', no_words=20, embedding_dim=400, batch_size=100):

        super(PairBisProcessor, self).__init__(pairs_filename, no_pairs_filename,
                                               docfreq_filename, w2v_filename, no_words,
                                               embedding_dim, batch_size)

        self.x1_bis = np.zeros((batch_size, no_words), dtype=theano.config.floatX)
        self.x2_bis = np.zeros((batch_size, no_words), dtype=theano.config.floatX)

    def process_batch(self):
        for i in xrange(0, self.batch_size, 2):
            pair = self.pairs_file.next().split(';')
            no_pair = self.no_pairs_file.next().split(';')

            pairA = pair[0].split()
            pairB = pair[1].split()
            no_pairA = no_pair[0].split()
            no_pairB = no_pair[1].split()

            dA = [0]*self.no_words
            dB = [0]*self.no_words
            nA = [0]*self.no_words
            nB = [0]*self.no_words

            for k in xrange(self.no_words):
                dA[k] = np.log(self.docfreqs[self.w.model.vocab[pairA[k]].index])
                dB[k] = np.log(self.docfreqs[self.w.model.vocab[pairB[k]].index])
                nA[k] = np.log(self.docfreqs[self.w.model.vocab[no_pairA[k]].index])
                nB[k] = np.log(self.docfreqs[self.w.model.vocab[no_pairB[k]].index])

            dA, pairA = zip(*sorted(zip(dA, pairA)))
            dB, pairB = zip(*sorted(zip(dB, pairB)))
            nA, no_pairA = zip(*sorted(zip(nA, no_pairA)))
            nB, no_pairB = zip(*sorted(zip(nB, no_pairB)))

            for j in xrange(self.no_words):
                self.x1[i, :, j] = self.w.get_vector(pairA[j])
                self.x2[i, :, j] = self.w.get_vector(pairB[j])
                self.x1_bis[i, j] = np.log(metrics.N_DOCUMENTS) - dA[j]
                self.x2_bis[i, j] = np.log(metrics.N_DOCUMENTS) - dB[j]
                self.x1[i+1, :, j] = self.w.get_vector(no_pairA[j])
                self.x2[i+1, :, j] = self.w.get_vector(no_pairB[j])
                self.x1_bis[i+1, j] = np.log(metrics.N_DOCUMENTS) - nA[j]
                self.x2_bis[i+1, j] = np.log(metrics.N_DOCUMENTS) - nB[j]

            self.y[i] = 0.0
            self.z[i] = -1.0
            self.y[i+1] = 1.0
            self.z[i+1] = 1.0

class unsortedPairProcessor(PairProcessor):
    def process_batch(self):
        for i in xrange(0, self.batch_size, 2):
            pair = self.pairs_file.next().split(';')
            no_pair = self.no_pairs_file.next().split(';')

            pairA = pair[0].split()
            pairB = pair[1].split()
            no_pairA = no_pair[0].split()
            no_pairB = no_pair[1].split()

            for j in xrange(self.no_words):
                self.x1[i, :, j] = self.w.get_vector(pairA[j])
                self.x2[i, :, j] = self.w.get_vector(pairB[j])
                self.x1[i+1, :, j] = self.w.get_vector(no_pairA[j])
                self.x2[i+1, :, j] = self.w.get_vector(no_pairB[j])

            self.y[i] = 0.0
            self.z[i] = -1.0
            self.y[i+1] = 1.0
            self.z[i+1] = 1.0

class lengthPairProcessor(PairProcessor):
    def __init__(self, pairs_filename='pairs.txt', no_pairs_filename='no_pairs.txt', docfreq_filename='docfreqs.npy',
                 w2v_filename='minimal', no_words=30, embedding_dim=400, batch_size=100):
        ## no_words is the maximum number of words allowed
        super(lengthPairProcessor, self).__init__(pairs_filename, no_pairs_filename,
                                               docfreq_filename, w2v_filename, no_words,
                                               embedding_dim, batch_size)

        self.l1 = np.zeros((batch_size), dtype=theano.config.floatX)
        self.l2 = np.zeros((batch_size), dtype=theano.config.floatX)

    def process_batch(self):
        for i in xrange(0, self.batch_size, 2):
            pair = self.pairs_file.next().split(';')
            no_pair = self.no_pairs_file.next().split(';')

            pairA = [k for k in pair[0].split() if self.w.exists_word(k)]
            pairB = [k for k in pair[1].split() if self.w.exists_word(k)]
            no_pairA = [k for k in no_pair[0].split() if self.w.exists_word(k)]
            no_pairB = [k for k in no_pair[1].split() if self.w.exists_word(k)]

            dA = [0]*len(pairA)
            dB = [0]*len(pairB)
            nA = [0]*len(no_pairA)
            nB = [0]*len(no_pairB)

            for k in xrange(len(pairA)):
                dA[k] = self.docfreqs[self.w.model.vocab[pairA[k]].index]
            for k in xrange(len(pairB)):
                dB[k] = self.docfreqs[self.w.model.vocab[pairB[k]].index]
            for k in xrange(len(no_pairA)):
                nA[k] = self.docfreqs[self.w.model.vocab[no_pairA[k]].index]
            for k in xrange(len(no_pairB)):
                nB[k] = self.docfreqs[self.w.model.vocab[no_pairB[k]].index]

            _, pairA = zip(*sorted(zip(dA, pairA)))
            _, pairB = zip(*sorted(zip(dB, pairB)))
            _, no_pairA = zip(*sorted(zip(nA, no_pairA)))
            _, no_pairB = zip(*sorted(zip(nB, no_pairB)))

            self.x1[i, :, :] = 0.0
            self.x2[i, :, :] = 0.0
            self.x1[i+1, :, :] = 0.0
            self.x2[i+1, :, :] = 0.0
            for j in xrange(len(pairA)):
                self.x1[i, :, j] = self.w.get_vector(pairA[j])
            for j in xrange(len(pairB)):
                self.x2[i, :, j] = self.w.get_vector(pairB[j])
            for j in xrange(len(no_pairA)):
                self.x1[i+1, :, j] = self.w.get_vector(no_pairA[j])
            for j in xrange(len(no_pairB)):
                self.x2[i+1, :, j] = self.w.get_vector(no_pairB[j])

            self.l1[i] = len(pairA) - 1
            self.l2[i] = len(pairB) - 1
            self.l1[i+1] = len(no_pairA) - 1
            self.l2[i+1] = len(no_pairB) - 1
            self.y[i] = 0.0
            self.z[i] = -1.0
            self.y[i+1] = 1.0
            self.z[i+1] = 1.0

class lengthTweetPairProcessor(PairProcessor):
    def __init__(self, pairs_filename='pairs.txt', no_pairs_filename='no_pairs.txt', docfreq_filename='docfreqs.npy',
                 w2v_filename='minimal', no_words=30, embedding_dim=400, batch_size=100, cutoff_function=None):
        ## no_words is the maximum number of words allowed
        super(lengthTweetPairProcessor, self).__init__(pairs_filename, no_pairs_filename,
                                               docfreq_filename, w2v_filename, no_words,
                                               embedding_dim, batch_size)

        self.l1 = np.zeros((batch_size), dtype=theano.config.floatX)
        self.l2 = np.zeros((batch_size), dtype=theano.config.floatX)
        self.cutoff_function = cutoff_function
        self.indices1 = np.zeros((no_words, batch_size), dtype=theano.config.floatX)
        self.indices2 = np.zeros((no_words, batch_size), dtype=theano.config.floatX)

    def process_batch(self):
        self.indices1[:, :] = 0.0
        self.indices2[:, :] = 0.0
        for i in xrange(0, self.batch_size, 2):
            pair = self.pairs_file.next().split(';')
            no_pair = self.no_pairs_file.next().split(';')

            pairA = [k for k in pair[0].split() if self.w.exists_word(k)]
            pairB = [k for k in pair[1].split() if self.w.exists_word(k)]
            no_pairA = [k for k in no_pair[0].split() if self.w.exists_word(k)]
            no_pairB = [k for k in no_pair[1].split() if self.w.exists_word(k)]

            dA = [0]*len(pairA)
            dB = [0]*len(pairB)
            nA = [0]*len(no_pairA)
            nB = [0]*len(no_pairB)

            for k in xrange(len(pairA)):
                dA[k] = self.docfreqs[self.w.model.vocab[pairA[k]].index]
            for k in xrange(len(pairB)):
                dB[k] = self.docfreqs[self.w.model.vocab[pairB[k]].index]
            for k in xrange(len(no_pairA)):
                nA[k] = self.docfreqs[self.w.model.vocab[no_pairA[k]].index]
            for k in xrange(len(no_pairB)):
                nB[k] = self.docfreqs[self.w.model.vocab[no_pairB[k]].index]

            _, pairA = zip(*sorted(zip(dA, pairA)))
            _, pairB = zip(*sorted(zip(dB, pairB)))
            _, no_pairA = zip(*sorted(zip(nA, no_pairA)))
            _, no_pairB = zip(*sorted(zip(nB, no_pairB)))

            self.x1[i, :, :] = 0.0
            self.x2[i, :, :] = 0.0
            self.x1[i+1, :, :] = 0.0
            self.x2[i+1, :, :] = 0.0
            for j in xrange(len(pairA)):
                self.x1[i, :, j] = self.w.get_vector(pairA[j])
            for j in xrange(len(pairB)):
                self.x2[i, :, j] = self.w.get_vector(pairB[j])
            for j in xrange(len(no_pairA)):
                self.x1[i+1, :, j] = self.w.get_vector(no_pairA[j])
            for j in xrange(len(no_pairB)):
                self.x2[i+1, :, j] = self.w.get_vector(no_pairB[j])

            self.l1[i] = len(pairA) - 1
            self.l2[i] = len(pairB) - 1
            self.l1[i+1] = len(no_pairA) - 1
            self.l2[i+1] = len(no_pairB) - 1
            self.y[i] = 0.0
            self.z[i] = -1.0
            self.y[i+1] = 1.0
            self.z[i+1] = 1.0

            self.indices1[0:self.l1[i]+1, i] = np.transpose(np.linspace(0,
                                                    (1 - self.cutoff_function(self.l1[i] + 1))*(self.no_words - 1), self.l1[i] + 1))
            self.indices2[0:self.l2[i]+1, i] = np.transpose(np.linspace(0,
                                                    (1 - self.cutoff_function(self.l2[i] + 1))*(self.no_words - 1), self.l2[i] + 1))
            self.indices1[0:self.l1[i+1]+1, i+1] = np.transpose(np.linspace(0,
                                                    (1 - self.cutoff_function(self.l1[i+1] + 1))*(self.no_words - 1), self.l1[i+1] + 1))
            self.indices2[0:self.l2[i+1]+1, i+1] = np.transpose(np.linspace(0,
                                                    (1 - self.cutoff_function(self.l2[i+1] + 1))*(self.no_words - 1), self.l2[i+1] + 1))


class lengthLinTweetPairProcessor(PairProcessor):
    def __init__(self, pairs_filename='pairs.txt', no_pairs_filename='no_pairs.txt', docfreq_filename='docfreqs.npy',
                 w2v_filename='minimal', no_words=30, embedding_dim=400, batch_size=100):
        ## no_words is the maximum number of words allowed
        super(lengthLinTweetPairProcessor, self).__init__(pairs_filename, no_pairs_filename,
                                               docfreq_filename, w2v_filename, no_words,
                                               embedding_dim, batch_size)

        self.l1 = np.zeros((batch_size), dtype=theano.config.floatX)
        self.l2 = np.zeros((batch_size), dtype=theano.config.floatX)
        self.indices1 = np.zeros((no_words, batch_size), dtype=theano.config.floatX)
        self.indices2 = np.zeros((no_words, batch_size), dtype=theano.config.floatX)

    def process_batch(self):
        self.indices1[:, :] = 0.0
        self.indices2[:, :] = 0.0
        for i in xrange(0, self.batch_size, 2):
            pair = self.pairs_file.next().split(';')
            no_pair = self.no_pairs_file.next().split(';')

            pairA = [k for k in pair[0].split() if self.w.exists_word(k)]
            pairB = [k for k in pair[1].split() if self.w.exists_word(k)]
            no_pairA = [k for k in no_pair[0].split() if self.w.exists_word(k)]
            no_pairB = [k for k in no_pair[1].split() if self.w.exists_word(k)]

            dA = [0]*len(pairA)
            dB = [0]*len(pairB)
            nA = [0]*len(no_pairA)
            nB = [0]*len(no_pairB)

            for k in xrange(len(pairA)):
                dA[k] = self.docfreqs[self.w.model.vocab[pairA[k]].index]
            for k in xrange(len(pairB)):
                dB[k] = self.docfreqs[self.w.model.vocab[pairB[k]].index]
            for k in xrange(len(no_pairA)):
                nA[k] = self.docfreqs[self.w.model.vocab[no_pairA[k]].index]
            for k in xrange(len(no_pairB)):
                nB[k] = self.docfreqs[self.w.model.vocab[no_pairB[k]].index]

            _, pairA = zip(*sorted(zip(dA, pairA)))
            _, pairB = zip(*sorted(zip(dB, pairB)))
            _, no_pairA = zip(*sorted(zip(nA, no_pairA)))
            _, no_pairB = zip(*sorted(zip(nB, no_pairB)))

            self.x1[i, :, :] = 0.0
            self.x2[i, :, :] = 0.0
            self.x1[i+1, :, :] = 0.0
            self.x2[i+1, :, :] = 0.0
            for j in xrange(len(pairA)):
                self.x1[i, :, j] = self.w.get_vector(pairA[j])
            for j in xrange(len(pairB)):
                self.x2[i, :, j] = self.w.get_vector(pairB[j])
            for j in xrange(len(no_pairA)):
                self.x1[i+1, :, j] = self.w.get_vector(no_pairA[j])
            for j in xrange(len(no_pairB)):
                self.x2[i+1, :, j] = self.w.get_vector(no_pairB[j])

            self.l1[i] = len(pairA) - 1
            self.l2[i] = len(pairB) - 1
            self.l1[i+1] = len(no_pairA) - 1
            self.l2[i+1] = len(no_pairB) - 1
            self.y[i] = 0.0
            self.z[i] = -1.0
            self.y[i+1] = 1.0
            self.z[i+1] = 1.0

            self.indices1[0:self.l1[i]+1, i] = np.transpose(np.linspace(0, self.l1[i], self.l1[i] + 1))
            self.indices2[0:self.l2[i]+1, i] = np.transpose(np.linspace(0, self.l2[i], self.l2[i] + 1))
            self.indices1[0:self.l1[i+1]+1, i+1] = np.transpose(np.linspace(0, self.l1[i+1], self.l1[i+1] + 1))
            self.indices2[0:self.l2[i+1]+1, i+1] =np.transpose(np.linspace(0, self.l2[i+1], self.l2[i+1] + 1))


class lengthSingleTweetPairProcessor(PairProcessor):
    def __init__(self, pairs_filename='pairs.txt', docfreq_filename='docfreqs.npy',
                 w2v_filename='minimal', no_words=30, embedding_dim=400, batch_size=100, cutoff_function=None):
        ## no_words is the maximum number of words allowed
        super(lengthSingleTweetPairProcessor, self).__init__(pairs_filename, None,
                                               docfreq_filename, w2v_filename, no_words,
                                               embedding_dim, batch_size)

        self.l1 = np.zeros((batch_size), dtype=theano.config.floatX)
        self.l2 = np.zeros((batch_size), dtype=theano.config.floatX)
        self.cutoff_function = cutoff_function
        self.indices1 = np.zeros((no_words, batch_size), dtype=theano.config.floatX)
        self.indices2 = np.zeros((no_words, batch_size), dtype=theano.config.floatX)

    def process_batch(self):
        self.indices1[:, :] = 0.0
        self.indices2[:, :] = 0.0
        for i in xrange(0, self.batch_size, 1):
            pair = self.pairs_file.next().split(';')

            pairA = [k for k in pair[0].split() if self.w.exists_word(k)]
            pairB = [k for k in pair[1].split() if self.w.exists_word(k)]

            dA = [0]*len(pairA)
            dB = [0]*len(pairB)

            for k in xrange(len(pairA)):
                dA[k] = self.docfreqs[self.w.model.vocab[pairA[k]].index]
            for k in xrange(len(pairB)):
                dB[k] = self.docfreqs[self.w.model.vocab[pairB[k]].index]

            _, pairA = zip(*sorted(zip(dA, pairA)))
            _, pairB = zip(*sorted(zip(dB, pairB)))

            self.x1[i, :, :] = 0.0
            self.x2[i, :, :] = 0.0
            for j in xrange(len(pairA)):
                self.x1[i, :, j] = self.w.get_vector(pairA[j])
            for j in xrange(len(pairB)):
                self.x2[i, :, j] = self.w.get_vector(pairB[j])

            self.l1[i] = len(pairA) - 1
            self.l2[i] = len(pairB) - 1
            self.y[i] = 0.0
            self.z[i] = -1.0

            self.indices1[0:self.l1[i]+1, i] = np.transpose(np.linspace(0,
                                                    (1 - self.cutoff_function(self.l1[i]+1))*(self.no_words - 1), self.l1[i]+1))
            self.indices2[0:self.l2[i]+1, i] = np.transpose(np.linspace(0,
                                                    (1 - self.cutoff_function(self.l2[i]+1))*(self.no_words - 1), self.l2[i]+1))