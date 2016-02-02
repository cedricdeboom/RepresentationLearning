"""
Similarity measures for wiki pairs
Based on w2v word list
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 1st"

import copy
import math
import numpy as np
import scipy.spatial.distance
import scipy.stats
from w2v import w2v

"""
CONSTANTS
"""
N_DOCUMENTS = 137964149

N_VAL = 1900000
N_TEST = 1500000
N_TRAIN = 1500000



"""
DISTANCES
"""
def cosine(A, B):
    return scipy.spatial.distance.cosine(A, B)

def euclidean(A, B):
    return scipy.spatial.distance.euclidean(A, B)

def braycurtis(A, B):
    return scipy.spatial.distance.braycurtis(A, B)

def minkowski3(A, B):
    return scipy.spatial.distance.minkowski(A, B, 3)

def minkowski4(A, B):
    return scipy.spatial.distance.minkowski(A, B, 4)

import NN_trained_weights as weights


"""
EVALUATION METRICS
"""
def tfidf(distance=cosine): #factory
    def tfidfInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on tf-idf
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py
        :return: cosine similarity between sentence a and sentence b based on tf-idf
        """
        wordsA = a.split()
        wordsB = b.split()
        wordsCommon = set.intersection(set(wordsA), set(wordsB))

        if len(wordsCommon) == 0:
            return 0.0

        sim = 0.0
        for word in wordsCommon:
            sim += wordsA.count(word) * wordsB.count(word) * pow(np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[word].index]), 2)

        den1 = 0.0
        for word in wordsA:
            den1 += pow(wordsA.count(word) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[word].index])), 2)
        den2 = 0.0
        for word in wordsB:
            den2 += pow(wordsB.count(word) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[word].index])), 2)

        return sim / np.sqrt(den1) / np.sqrt(den2)

    return tfidfInner

def mean(distance=cosine): #factory
    def meanInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on averaging word embeddings
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on averaging word embeddings
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, len(wordsA)):
            vecA += w2v_model.get_vector(wordsA[i])
        for i in xrange(1, len(wordsB)):
            vecB += w2v_model.get_vector(wordsB[i])

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        return 1 - distance(vecA, vecB)

    return meanInner

def WMean(distance=cosine): #factory
    def WMeanInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on averaging word embeddings and weighted by document frequencies
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: weighted cosine similarity between sentence a and sentence b based on averaging word embeddings
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0])) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsA[0]].index]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0])) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsB[0]].index]))
        for i in xrange(1, len(wordsA)):
            vecA += w2v_model.get_vector(wordsA[i]) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsA[i]].index]))
        for i in xrange(1, len(wordsB)):
            vecB += w2v_model.get_vector(wordsB[i]) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsB[i]].index]))

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        if distance == cosine:
            return 1 - distance(vecA, vecB)
        #return (np.log(N_DOCUMENTS) - distance(vecA, vecB)) / np.log(N_DOCUMENTS)
        return 1.0 - distance(vecA, vecB)

    return WMeanInner


def WMeanIdf(distance=euclidean, idf_weights=None): #factory
    def WMeanIdfInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate similarity between two sentences based on averaging word embeddings and weighted by normalized 'average' document frequencies
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: averaged weighted cosine similarity between sentence a and sentence b based on averaging word embeddings
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        if np.isnan(idf_weights[len(wordsA), 0]):
            temp = np.zeros(len(wordsA))
            for i in xrange(len(wordsA)):
                temp[i] = (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsA[i]].index]))
            temp /= np.max(temp)
            vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0])) * temp[0]
            for i in xrange(1, len(wordsA)):
                vecA += w2v_model.get_vector(wordsA[i]) * temp[i]
        if np.isnan(idf_weights[len(wordsB), 0]):
            temp = np.zeros(len(wordsB))
            for i in xrange(len(wordsB)):
                temp[i] = (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsB[i]].index]))
            temp /= np.max(temp)
            vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0])) * temp[0]
            for i in xrange(1, len(wordsB)):
                vecB += w2v_model.get_vector(wordsB[i]) * temp[i]
        if 'vecA' not in locals():
            vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0])) * idf_weights[len(wordsA), 0]
            for i in xrange(1, len(wordsA)):
                vecA += w2v_model.get_vector(wordsA[i]) * idf_weights[len(wordsA), i]
        if 'vecB' not in locals():
            vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0])) * idf_weights[len(wordsB), 0]
            for i in xrange(1, len(wordsB)):
                vecB += w2v_model.get_vector(wordsB[i]) * idf_weights[len(wordsB), i]

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        if distance == cosine:
            return 1 - distance(vecA, vecB)
        return 1.0 - distance(vecA, vecB)

    return WMeanIdfInner


def meanTop(distance=cosine, top=1.0/3.0): #factory
    def meanTopInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on averaging word embeddings with top-k doc frequencies
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: cosine similarity between sentence a and sentence b based on averaging word embeddings with top-k doc frequencies
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, int(len(wordsA) * top)):
            vecA += w2v_model.get_vector(wordsA[i])
        for i in xrange(1, int(len(wordsB) * top)):
            vecB += w2v_model.get_vector(wordsB[i])

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        return 1 - distance(vecA, vecB)

    return meanTopInner

def NNMean(distance=euclidean, length=20): #factory
    WEIGHTS = weights.get_weights_variable_name(distance, length)

    def NNTopInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on averaging word embeddings with NN weights
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: similarity between sentence a and sentence b based on averaging word embeddings with NN weights
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        if length == 'R':
            a = np.linspace(0.0, 34.0, len(wordsA))
            b = np.linspace(0.0, 34.0, len(wordsB))
            WEIGHTS_A = WEIGHTS[0] * (a ** 2) + WEIGHTS[1] * a + WEIGHTS[2]
            WEIGHTS_B = WEIGHTS[0] * (b ** 2) + WEIGHTS[1] * b + WEIGHTS[2]
            WEIGHTS_A /= WEIGHTS_A[0]
            WEIGHTS_B /= WEIGHTS_B[0]

            vecA = WEIGHTS_A[0] * copy.deepcopy(w2v_model.get_vector(wordsA[0]))
            vecB = WEIGHTS_B[0] * copy.deepcopy(w2v_model.get_vector(wordsB[0]))
            for i in xrange(1, int(len(wordsA))):
                vecA += WEIGHTS_A[i] * w2v_model.get_vector(wordsA[i])
            for i in xrange(1, int(len(wordsB))):
                vecB += WEIGHTS_B[i] * w2v_model.get_vector(wordsB[i])
        else:
            vecA = WEIGHTS[0] * copy.deepcopy(w2v_model.get_vector(wordsA[0]))
            vecB = WEIGHTS[0] * copy.deepcopy(w2v_model.get_vector(wordsB[0]))
            for i in xrange(1, int(len(wordsA))):
                vecA += WEIGHTS[i] * w2v_model.get_vector(wordsA[i])
            for i in xrange(1, int(len(wordsB))):
                vecB += WEIGHTS[i] * w2v_model.get_vector(wordsB[i])

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        ret = 1.0 - distance(vecA, vecB)
        if distance == cosine:
            ret = 1.0 - distance(vecA, vecB)/2.0
        return ret

    return NNTopInner

def NNExponMean(distance=euclidean, length='T'): #factory
    WEIGHTS = weights.get_weights_variable_name(distance, length)

    def cutoff(x):
        #return 0.92258 * np.exp(-0.27144 * x) + 0.09834
        return 0.0

    def NNTopInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on averaging word embeddings with NN weights
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: similarity between sentence a and sentence b based on averaging word embeddings with NN weights
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        if length == 'T':
            a = np.transpose(np.linspace(0, (1 - cutoff(len(wordsA)))*34.0, len(wordsA)))
            b = np.transpose(np.linspace(0, (1 - cutoff(len(wordsB)))*34.0, len(wordsB)))
            WEIGHTS_A = WEIGHTS[0] * np.exp(WEIGHTS[1] * a) + WEIGHTS[2]
            WEIGHTS_B = WEIGHTS[0] * np.exp(WEIGHTS[1] * b) + WEIGHTS[2]

            vecA = WEIGHTS_A[0] * copy.deepcopy(w2v_model.get_vector(wordsA[0]))
            vecB = WEIGHTS_B[0] * copy.deepcopy(w2v_model.get_vector(wordsB[0]))
            for i in xrange(1, int(len(wordsA))):
                vecA += WEIGHTS_A[i] * w2v_model.get_vector(wordsA[i])
            for i in xrange(1, int(len(wordsB))):
                vecB += WEIGHTS_B[i] * w2v_model.get_vector(wordsB[i])
            vecA /= len(wordsA)
            vecB /= len(wordsB)
        else:
            vecA = WEIGHTS[0] * copy.deepcopy(w2v_model.get_vector(wordsA[0]))
            vecB = WEIGHTS[0] * copy.deepcopy(w2v_model.get_vector(wordsB[0]))
            for i in xrange(1, int(len(wordsA))):
                vecA += WEIGHTS[i] * w2v_model.get_vector(wordsA[i])
            for i in xrange(1, int(len(wordsB))):
                vecB += WEIGHTS[i] * w2v_model.get_vector(wordsB[i])
            vecA /= len(wordsA)
            vecB /= len(wordsB)

        ret = 1.0 - distance(vecA, vecB)
        if distance == cosine:
            ret = 1.0 - distance(vecA, vecB)/2.0
        return ret

    return NNTopInner


def NNVarMean(distance=euclidean, length='V'): #factory
    WEIGHTS = np.asarray(weights.get_weights_variable_name(distance, length))

    def cutoff(x):
        #return 0.92258 * np.exp(-0.27144 * x) + 0.09834
        return 0.0

    def NNVarInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate similarity between two sentences based on averaging word embeddings with NN weights
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: similarity between sentence a and sentence b based on averaging word embeddings with NN weights
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        if length == 'V':
            a = np.linspace(0.0, (1 - cutoff(len(wordsA)))*34.0, len(wordsA))
            b = np.linspace(0.0, (1 - cutoff(len(wordsB)))*34.0, len(wordsB))

            a_high = np.ceil(a).astype(np.int8)
            a_low = np.floor(a).astype(np.int8)
            af_high = WEIGHTS[a_high]
            af_low = WEIGHTS[a_low]
            af = ((af_high - af_low) * (a - a_low) / (a_high - a_low + 1E-5)) + af_low
            b_high = np.ceil(b).astype(np.int8)
            b_low = np.floor(b).astype(np.int8)
            bf_high = WEIGHTS[b_high]
            bf_low = WEIGHTS[b_low]
            bf = ((bf_high - bf_low) * (b - b_low) / (b_high - b_low + 1E-5)) + bf_low

            vecA = af[0] * copy.deepcopy(w2v_model.get_vector(wordsA[0]))
            vecB = bf[0] * copy.deepcopy(w2v_model.get_vector(wordsB[0]))
            for i in xrange(1, int(len(wordsA))):
                vecA += af[i] * w2v_model.get_vector(wordsA[i])
            for i in xrange(1, int(len(wordsB))):
                vecB += bf[i] * w2v_model.get_vector(wordsB[i])
        else:
            print 'Length must be \'V\'\nNow exiting...'
            import sys
            sys.exit(0)

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        ret = 1.0 - distance(vecA, vecB)
        if distance == cosine:
            ret = 1.0 - distance(vecA, vecB)/2.0
        return ret

    return NNVarInner


def NNVarLinMean(distance=euclidean, length='L'): #factory
    WEIGHTS = np.asarray(weights.get_weights_variable_name(distance, length))

    def NNVarLinInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate similarity between two sentences based on averaging word embeddings with NN weights
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: similarity between sentence a and sentence b based on averaging word embeddings with NN weights
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        if length == 'L':
            a = np.linspace(0.0, len(wordsA) - 1, len(wordsA)).astype('int8')
            b = np.linspace(0.0, len(wordsB) - 1, len(wordsB)).astype('int8')

            af = WEIGHTS[a]
            bf = WEIGHTS[b]

            vecA = af[0] * copy.deepcopy(w2v_model.get_vector(wordsA[0]))
            vecB = bf[0] * copy.deepcopy(w2v_model.get_vector(wordsB[0]))
            for i in xrange(1, int(len(wordsA))):
                vecA += af[i] * w2v_model.get_vector(wordsA[i])
            for i in xrange(1, int(len(wordsB))):
                vecB += bf[i] * w2v_model.get_vector(wordsB[i])
        else:
            print 'Length must be \'L\'\nNow exiting...'
            import sys
            sys.exit(0)

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        ret = 1.0 - distance(vecA, vecB)
        if distance == cosine:
            ret = 1.0 - distance(vecA, vecB)/2.0
        return ret

    return NNVarLinInner


def NNVarMeanWMean(distance=euclidean, length='V'): #factory
    WEIGHTS = np.asarray(weights.get_weights_variable_name(distance, length))

    def cutoff(x):
        #return 0.92258 * np.exp(-0.27144 * x) + 0.09834
        return 0.0

    def NNVarWMeanInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate similarity between two sentences based on averaging word embeddings with NN weights
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: similarity between sentence a and sentence b based on averaging word embeddings with NN weights
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        if length == 'V':
            a = np.linspace(0.0, (1 - cutoff(len(wordsA)))*34.0, len(wordsA))
            b = np.linspace(0.0, (1 - cutoff(len(wordsB)))*34.0, len(wordsB))

            a_high = np.ceil(a).astype(np.int8)
            a_low = np.floor(a).astype(np.int8)
            af_high = WEIGHTS[a_high]
            af_low = WEIGHTS[a_low]
            af = ((af_high - af_low) * (a - a_low) / (a_high - a_low + 1E-5)) + af_low
            b_high = np.ceil(b).astype(np.int8)
            b_low = np.floor(b).astype(np.int8)
            bf_high = WEIGHTS[b_high]
            bf_low = WEIGHTS[b_low]
            bf = ((bf_high - bf_low) * (b - b_low) / (b_high - b_low + 1E-5)) + bf_low

            vecA_n = af[0] * copy.deepcopy(w2v_model.get_vector(wordsA[0]))
            vecB_n = bf[0] * copy.deepcopy(w2v_model.get_vector(wordsB[0]))
            for i in xrange(1, int(len(wordsA))):
                vecA_n += af[i] * w2v_model.get_vector(wordsA[i])
            for i in xrange(1, int(len(wordsB))):
                vecB_n += bf[i] * w2v_model.get_vector(wordsB[i])

            vecA_w = copy.deepcopy(w2v_model.get_vector(wordsA[0])) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsA[0]].index]))
            vecB_w = copy.deepcopy(w2v_model.get_vector(wordsB[0])) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsB[0]].index]))
            for i in xrange(1, len(wordsA)):
                vecA_w += w2v_model.get_vector(wordsA[i]) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsA[i]].index]))
            for i in xrange(1, len(wordsB)):
                vecB_w += w2v_model.get_vector(wordsB[i]) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsB[i]].index]))
        else:
            print 'Length must be \'V\'\nNow exiting...'
            import sys
            sys.exit(0)

        vecA_n /= len(wordsA)
        vecB_n /= len(wordsB)
        vecA_w /= len(wordsA)
        vecB_w /= len(wordsB)

        vecA = (vecA_n + vecA_w) / 2.0
        vecB = (vecB_n + vecB_w) / 2.0

        ret = 1.0 - distance(vecA, vecB)
        if distance == cosine:
            ret = 1.0 - distance(vecA, vecB)/2.0
        return ret

    return NNVarWMeanInner


def WMeanTop(distance=cosine, top=1.0/3.0): #factory
    def WMeanTopInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on averaging word embeddings with top-k doc frequencies and weighted by document frequencies
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: cosine similarity between sentence a and sentence b based on averaging word embeddings with top-k doc frequencies and weighted by document frequencies
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0])) * (np.log(N_DOCUMENTS) - np.log(fA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0])) * (np.log(N_DOCUMENTS) - np.log(fB[0]))
        for i in xrange(1, int(len(wordsA) * top)):
            vecA += w2v_model.get_vector(wordsA[i]) * (np.log(N_DOCUMENTS) - np.log(fA[i]))
        for i in xrange(1, int(len(wordsB) * top)):
            vecB += w2v_model.get_vector(wordsB[i]) * (np.log(N_DOCUMENTS) - np.log(fB[i]))

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        if distance == cosine:
            return 1 - distance(vecA, vecB)
        return (np.log(N_DOCUMENTS) - distance(vecA, vecB)) / np.log(N_DOCUMENTS)

    return WMeanTopInner

def NNWMean(distance=euclidean, length=20): #factory
    WEIGHTS = weights.get_weights_variable_name(distance, length) + '_IDF'
    def NNWMeanTopInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on averaging word embeddings with top-k doc frequencies and weighted by document frequencies + IDF_WEIGHTS
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: cosine similarity between sentence a and sentence b based on averaging word embeddings with top-k doc frequencies and weighted by document frequencies + IDF_WEIGHTS
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0])) * (np.log(N_DOCUMENTS) - np.log(fA[0])) * WEIGHTS[0]
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0])) * (np.log(N_DOCUMENTS) - np.log(fB[0])) * WEIGHTS[0]
        for i in xrange(1, len(wordsA)):
            vecA += w2v_model.get_vector(wordsA[i]) * (np.log(N_DOCUMENTS) - np.log(fA[i])) * WEIGHTS[i]
        for i in xrange(1, len(wordsB)):
            vecB += w2v_model.get_vector(wordsB[i]) * (np.log(N_DOCUMENTS) - np.log(fB[i])) * WEIGHTS[i]

        vecA /= len(wordsA)
        vecB /= len(wordsB)

        if distance == cosine:
            return 1 - distance(vecA, vecB)
        return (np.log(N_DOCUMENTS) - distance(vecA, vecB)) / np.log(N_DOCUMENTS)

    return NNWMeanTopInner

def maxTop(distance=cosine, top=1.0/3.0): #factory
    def maxTopInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on maximizing word embeddings with top-k doc frequencies
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: cosine similarity between sentence a and sentence b based on maximizing word embeddings with top-k doc frequencies
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, int(len(wordsA) * top)):
            src = w2v_model.get_vector(wordsA[i])
            np.copyto(vecA, src, where=(src > vecA))
        for i in xrange(1, int(len(wordsB) * top)):
            src = w2v_model.get_vector(wordsB[i])
            np.copyto(vecB, src, where=(src > vecB))

        if distance == euclidean:
            return 1.0 - distance(vecA, vecB)/2.0
        return 1 - distance(vecA, vecB)
    return maxTopInner

def randomMinMax(distance=cosine): #factory
    def randomMinMaxInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on maximizing or minimizing (randomly) word embeddings in each dimension
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on maximizing or minimizing (randomly) word embeddings in each dimension
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecAmax = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecAmin = copy.deepcopy(vecAmax)
        vecBmax = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        vecBmin = copy.deepcopy(vecBmax)
        for i in xrange(1, len(wordsA)):
            src = w2v_model.get_vector(wordsA[i])
            np.copyto(vecAmax, src, where=(src > vecAmax))
            np.copyto(vecAmin, src, where=(src < vecAmin))
        for i in xrange(1, len(wordsB)):
            src = w2v_model.get_vector(wordsB[i])
            np.copyto(vecBmax, src, where=(src > vecBmax))
            np.copyto(vecBmin, src, where=(src < vecBmin))

        resA = np.zeros(vecAmax.shape)
        resB = np.zeros(vecBmax.shape)
        for i in xrange(len(resA)):
            if np.random.randint(2) == 1:
                resA[i] = vecAmax[i]
                resB[i] = vecBmax[i]
            else:
                resA[i] = vecAmin[i]
                resB[i] = vecBmin[i]

        if distance == euclidean:
            return 1.0 - distance(resA, resB)/5.0
        return 1 - distance(resA, resB)
    return randomMinMaxInner


def max(distance=cosine): #factory
    def maxInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on maximizing word embeddings in each dimension
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on maximizing word embeddings in each dimension
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, len(wordsA)):
            src = w2v_model.get_vector(wordsA[i])
            np.copyto(vecA, src, where=(src > vecA))
        for i in xrange(1, len(wordsB)):
            src = w2v_model.get_vector(wordsB[i])
            np.copyto(vecB, src, where=(src > vecB))

        return 1 - distance(vecA, vecB)
    return maxInner

def minMax(distance=cosine): #factory
    def minMaxInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on maximizing and minimizing word embeddings in each dimension
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on maximizing and minimizing word embeddings in each dimension
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecAmax = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecAmin = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecBmax = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        vecBmin = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, len(wordsA)):
            src = w2v_model.get_vector(wordsA[i])
            np.copyto(vecAmax, src, where=(src > vecAmax))
            np.copyto(vecAmin, src, where=(src < vecAmin))
        for i in xrange(1, len(wordsB)):
            src = w2v_model.get_vector(wordsB[i])
            np.copyto(vecBmax, src, where=(src > vecBmax))
            np.copyto(vecBmin, src, where=(src < vecBmin))

        vecA = np.hstack((vecAmax, vecAmin))
        vecB = np.hstack((vecBmax, vecBmin))

        if distance == euclidean:
            return 1.0 - distance(vecA, vecB)/3.0
        return 1 - distance(vecA, vecB)
    return minMaxInner

def minMaxTop(distance=cosine, top=1.0/3.0): #factory
    def minMaxTopInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on maximizing and minimizing word embeddings in each dimension
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on maximizing and minimizing word embeddings in each dimension
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        vecAmax = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecAmin = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecBmax = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        vecBmin = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, int(len(wordsA) * top)):
            src = w2v_model.get_vector(wordsA[i])
            np.copyto(vecAmax, src, where=(src > vecAmax))
            np.copyto(vecAmin, src, where=(src < vecAmin))
        for i in xrange(1, int(len(wordsB) * top)):
            src = w2v_model.get_vector(wordsB[i])
            np.copyto(vecBmax, src, where=(src > vecBmax))
            np.copyto(vecBmin, src, where=(src < vecBmin))

        vecA = np.hstack((vecAmax, vecAmin))
        vecB = np.hstack((vecBmax, vecBmin))

        if distance == euclidean:
            return 1.0 - distance(vecA, vecB)/6.0
        return 1 - distance(vecA, vecB)
    return minMaxTopInner

def WMax(distance=cosine): #factory
    def WMaxInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on maximizing word embeddings in each dimension weighed with idf
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on maximizing word embeddings in each dimension weighed with idf
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0])) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsA[0]].index]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0])) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsB[0]].index]))
        for i in xrange(1, len(wordsA)):
            src = w2v_model.get_vector(wordsA[i]) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsA[i]].index]))
            np.copyto(vecA, src, where=(src > vecA))
        for i in xrange(1, len(wordsB)):
            src = w2v_model.get_vector(wordsB[i]) * (np.log(N_DOCUMENTS) - np.log(docfreqs[w2v_model.model.vocab[wordsB[i]].index]))
            np.copyto(vecB, src, where=(src > vecB))

        if distance == cosine:
            return 1 - distance(vecA, vecB)
        return (np.log(N_DOCUMENTS) - distance(vecA, vecB)) / np.log(N_DOCUMENTS)

    return WMaxInner

def min(distance=cosine): #factory
    def minInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on minimizing word embeddings in each dimension
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on minimizing word embeddings in each dimension
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, len(wordsA)):
            src = w2v_model.get_vector(wordsA[i])
            np.copyto(vecA, src, where=(src < vecA))
        for i in xrange(1, len(wordsB)):
            src = w2v_model.get_vector(wordsB[i])
            np.copyto(vecB, src, where=(src < vecB))

        return 1 - distance(vecA, vecB)
    return minInner

def extreme(distance=cosine): #factory
    def extremeInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on extremes in word embeddings in each dimension
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on extremes in word embeddings in each dimensions
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, len(wordsA)):
            src = w2v_model.get_vector(wordsA[i])
            np.copyto(vecA, src, where=(abs(src) > abs(vecA)))
        for i in xrange(1, len(wordsB)):
            src = w2v_model.get_vector(wordsB[i])
            np.copyto(vecB, src, where=(abs(src) > abs(vecB)))

        return 1 - distance(vecA, vecB)
    return extremeInner

def extremeTop(distance=cosine, top=1.0/3.0): #factory
    def extremeTopInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on extremizing word embeddings with top-k doc frequencies
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (cannot be None)
        :return: cosine similarity between sentence a and sentence b based on extremizing word embeddings with top-k doc frequencies
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        fA = [0]*len(wordsA)
        fB = [0]*len(wordsB)

        for i in xrange(len(wordsA)):
            fA[i] = docfreqs[w2v_model.model.vocab[wordsA[i]].index]
        for i in xrange(len(wordsB)):
            fB[i] = docfreqs[w2v_model.model.vocab[wordsB[i]].index]

        fA, wordsA = zip(*sorted(zip(fA, wordsA)))
        fB, wordsB = zip(*sorted(zip(fB, wordsB)))

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, int(len(wordsA) * top)):
            src = w2v_model.get_vector(wordsA[i])
            np.copyto(vecA, src, where=(abs(src) > abs(vecA)))
        for i in xrange(1, int(len(wordsB) * top)):
            src = w2v_model.get_vector(wordsB[i])
            np.copyto(vecB, src, where=(abs(src) > abs(vecA)))

        return 1 - distance(vecA, vecB)
    return extremeTopInner

def concat(distance=cosine): #factory
    def concatInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate cosine similarity between two sentences based on stacking word embeddings
        Sentences a and b should contain the same number of words!
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: cosine similarity between sentence a and sentence b based on stacking word embeddings
        """
        wordsA = a.split()
        wordsB = b.split()

        if len(wordsA) < 1 or len(wordsB) < 1:
            return 0.0

        vecA = copy.deepcopy(w2v_model.get_vector(wordsA[0]))
        vecB = copy.deepcopy(w2v_model.get_vector(wordsB[0]))
        for i in xrange(1, len(wordsA)):
            vecA = np.hstack((vecA, w2v_model.get_vector(wordsA[i])))
        for i in xrange(1, len(wordsB)):
            vecB = np.hstack((vecB, w2v_model.get_vector(wordsB[i])))

        if len(vecA) != len(vecB):
            raise ValueError('Dimensions do not match!')

        return 1 - distance(vecA, vecB)
    return concatInner


def PCA(distance=cosine, dictionary=None, PCA_matrix=None): #factory
    def PCAInner(a='', b='', w2v_model=None, docfreqs=None):
        """
        Calculate similarity between two sentences based on PCA embeddings
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: similarity between sentence a and sentence b based on PCA embeddings
        """
        wordsA = a.split()
        wordsB = b.split()

        vecA = np.zeros(PCA_matrix.shape[1])
        vecB = np.zeros(PCA_matrix.shape[1])
        nA = 0
        nB = 0

        for word in wordsA:
            index = dictionary.token2id.get(word, -1)
            if index >= 0:
                vecA += PCA_matrix[index]
                nA += 1
        for word in wordsB:
            index = dictionary.token2id.get(word, -1)
            if index >= 0:
                vecB += PCA_matrix[index]
                nB += 1

        if nA == 0 or nB == 0:
            return 0.0

        vecA /= nA
        vecB /= nB

        if distance == euclidean:
            return 1 - (distance(vecA, vecB) / 250000.0)
        return 1 - distance(vecA, vecB)
    return PCAInner


def LDA(distance=cosine, dictionary=None, LDA_model=None): #factory
    def LDAInner(a='', b='', w2v_model=None, docfreqs=None, batch=False):
        """
        Calculate similarity between two sentences based on LDA
        :param a: first sentence
        :param b: second sentence
        :param w2v_model: word2vec model used for vocabulary lookup
        :param docfreqs: document frequency table, as calculated by docfreq.py (can be None)
        :return: similarity between sentence a and sentence b based on LDA
        """

        if batch:
            inferred = LDA_model.model.inference(a)[0]
            ret = np.zeros(len(inferred) / 2)

            for i in xrange(0, len(inferred), 2):
                if distance == euclidean:
                    ret[i/2] = 1 - (distance(inferred[i], inferred[i+1]) / 4000.0)
                else:
                    ret[i/2] = 1 - distance(inferred[i], inferred[i+1])

            return ret
        else:
            wordsA = dictionary.doc2bow(a.split())
            wordsB = dictionary.doc2bow(b.split())
            return (wordsA, wordsB)


    return LDAInner



"""
PERFORMANCE METRICS
"""
def optimal_error_rate(no_pairs, pairs):
    """
    Calculate the optimal error rate for using cosine similarity information
    :param no_pairs: numpy array of cosine similarity values of different non pairs of text
    :param pairs: numpy array of cosine similarity values of different pairs of text
    :return: (optimal error rate, optimal split point)
    """
    error = 1.0
    split = 0
    for sweep in np.linspace(0, 1.0, num=1001):
        #number1 = float(len(np.where(no_pairs > sweep)[0])) / len(no_pairs)
        #number2 = float(len(np.where(pairs <= sweep)[0])) / len(pairs)
        number1 = (no_pairs > sweep).sum()
        number2 = (pairs <= sweep).sum()
        current_error = float(number1 + number2) / (len(pairs) + len(no_pairs))
        if current_error <= error:
            error = current_error
            split = sweep
    return (error, split)


def error_rate(no_pairs, pairs, split):
    """
    Calculate the error rate for using cosine similarity information and given split point
    :param no_pairs: numpy array of cosine similarity values of different non pairs of text
    :param pairs: numpy array of cosine similarity values of different pairs of text
    :return: optimal error rate
    """
    number1 = float(len(np.where(no_pairs > split)[0])) / len(no_pairs)
    number2 = float(len(np.where(pairs <= split)[0])) / len(pairs)
    error = (number1 + number2) / 2.0
    return error


def prf(no_pairs, pairs, split):
    """
    Calculate precision, recall and F1 cosine similarity information and given split point
    :param no_pairs: numpy array of cosine similarity values of different non pairs of text
    :param pairs: numpy array of cosine similarity values of different pairs of text
    :return: p, r, f
    """
    TP = float(len(np.where(pairs > split)[0]))
    FP = float(len(np.where(pairs <= split)[0]))
    FN = float(len(np.where(no_pairs > split)[0]))

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f = 2 * p * r / (p + r)

    return p, r, f


def binom_test(no_pairs_A, pairs_A, no_pairs_B, pairs_B, splitA, splitB):
    """
    Calculate binomial test statistic for cosine similarity information and given split point
    :param no_pairs: numpy array of cosine similarity values of different non pairs of text
    :param pairs: numpy array of cosine similarity values of different pairs of text
    :return: p
    """
    number1 = (no_pairs_A > splitA).sum()
    number2 = (pairs_A <= splitA).sum()
    hypothesis = float(number1 + number2) / (len(no_pairs_A) + len(pairs_A))

    successes = (no_pairs_B > splitB).sum() + (pairs_B <= splitB).sum()
    trials = len(pairs_B) + len(no_pairs_B)
    p = scipy.stats.binom_test(successes, trials, hypothesis)
    return p


def bootstrap_test(no_pairs_A, pairs_A, no_pairs_B, pairs_B, splitA, splitB, repeat=10000):
    """
    Calculate bootstrap test statistic for cosine similarity information and given split point
    :return: p
    """
    set1 = no_pairs_A > splitA
    set2 = pairs_A <= splitA
    set3 = no_pairs_B > splitB
    set4 = pairs_B <= splitB
    observations = np.hstack((set1, set2, set3, set4))
    n = len(set1) + len(set2)
    m = len(set3) + len(set4)

    t_star = (float(set1.sum() + set2.sum()) / n) - (float(set3.sum() + set4.sum()) / m)

    t = np.zeros(repeat)
    for i in xrange(repeat):
        sample = np.random.choice(observations, len(observations), replace=True)
        x_star = np.mean(sample[0:n])
        y_star = np.mean(sample[n:n+m])
        t[i] = x_star - y_star

    p = float((t > t_star).sum()) / repeat
    return p

def KL_divergence(no_pairs, pairs):
    """
    Calculate the Kullback-Leibler divergence between pairs and non-pairs distributions (no need to normalize)
    :param no_pairs: numpy array of cosine similarity values of different non pairs of text
    :param pairs: numpy array of cosine similarity values of different pairs of text
    :return: KL divergence
    """
    h_p = np.histogram(pairs, bins=10000, range=(0.0, 1.00))[0].astype(float) + 1.0
    h_np = np.histogram(no_pairs, bins=10000, range=(0.0, 1.00))[0].astype(float) + 1.0

    return scipy.stats.entropy(h_p, h_np)

def JS_divergence(no_pairs, pairs):
    """
    Calculate the Jensen-Shannon divergence between pairs and non-pairs distributions (no need to normalize)
    :param no_pairs: numpy array of cosine similarity values of different non pairs of text
    :param pairs: numpy array of cosine similarity values of different pairs of text
    :return: Jensen-Shannon divergence
    """
    h_p = np.histogram(pairs, bins=10000, range=(-1.0, 1.50))[0].astype(float) + 1.0
    h_np = np.histogram(no_pairs, bins=10000, range=(-1.0, 1.50))[0].astype(float) + 1.0

    h_p /= np.sum(h_p)
    h_np /= np.sum(h_np)

    import warnings
    warnings.filterwarnings("ignore", category = RuntimeWarning)

    d1 = h_p*np.log2(2*h_p/(h_p+h_np))
    d2 = h_np*np.log2(2*h_np/(h_p+h_np))
    d1[np.isnan(d1)] = 0
    d2[np.isnan(d2)] = 0
    d = 0.5*np.sum(d1+d2)
    return d

def hellinger_distance(no_pairs, pairs):
    """
    Calculate the hellinger distance between pairs and non-pairs distributions (no need to normalize)
    :param no_pairs: numpy array of cosine similarity values of different non pairs of text
    :param pairs: numpy array of cosine similarity values of different pairs of text
    :return: hellinger distance
    """
    h_p = np.histogram(pairs, bins=10000, range=(0.0, 1.00))[0].astype(float) + 1.0
    h_np = np.histogram(no_pairs, bins=10000, range=(0.0, 1.00))[0].astype(float) + 1.0

    h_p /= np.sum(h_p)
    h_np /= np.sum(h_np)

    import warnings
    warnings.filterwarnings("ignore", category = RuntimeWarning)

    h_p_squared = np.sqrt(h_p)
    h_np_squared = np.sqrt(h_np)
    d = (1.0/np.sqrt(2)) * euclidean(h_p_squared, h_np_squared)
    return d


def kurtosis(data):
    """
    Calculate kurtosis (no need to normalize)
    :param data: numpy array
    :return: kurtosis (Fischer)
    """
    data = data.astype(float)
    return scipy.stats.kurtosis(data, axis=None)

def skew(data):
    """
    Calculate skew (no need to normalize)
    :param data: numpy array
    :return: skew
    """
    data = data.astype(float)
    return scipy.stats.skew(data, axis=None)

def variance(data):
    """
    Calculate variance (no need to normalize)
    :param data: numpy array
    :return: variance
    """
    data = data.astype(float)
    return np.var(data)




if __name__ == '__main__':
    # f = open('../data/docfreq.npy')
    # docfreqs = np.load(f)
    # f.close()
    #
    # w = w2v()
    # w.load_minimal('../data/minimal')
    #
    # a = 'in the middle of the 0rd century bc an independent'
    # b = 'some banks offer alternative debit card facilities to their customers'
    #
    # print maxCosine(a, b, w, docfreqs)
    B = np.linspace(0.4, 1.0, num=101)
    A = np.linspace(0.0, 0.6, num=101)
    error, split = optimal_error_rate(A, B)
    print error
    print split