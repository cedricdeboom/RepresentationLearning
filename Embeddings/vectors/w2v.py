"""
Word2vec processing
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 March 19th"


import numpy as np
import gensim
import cPickle
import logging
import os


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class dataIterator():
    def __init__(self, folder='.'):
        self.folder = folder
    def __iter__(self):
        for fname in os.listdir(self.folder):
            if not fname.endswith('.txt'):
                continue
            for line in open(os.path.join(self.folder, fname)):
                yield line.split()


class w2v():
    def __init__(self):
        self.model = None
    def train(self, folder='.', size=400, window=5, sample=1e-4, workers=4, min_count=5, negative=10):
        data = dataIterator(folder)
        self.model = gensim.models.Word2Vec(data, size=size, window=window, sample=sample, workers=workers, min_count=min_count, negative=negative, hs=0)
    def save(self, filename):
        if self.model is not None:
            logging.log(0, 'Saving w2v model...')
            self.model.init_sims(replace=True)
            self.model.save(filename)
            logging.log(0, 'Done saving w2v model')
        else:
            logging.log(1, 'No model present! Cannot save without model.')
    def save_minimal(self, filename):
        if self.model is not None:
            logging.log(0, 'Saving w2v minimal model...')
            f = open(filename + '.1.npy', 'wb')
            np.save(f, self.model.syn0)
            f.close()
            f = open(filename + '.2.dump', 'wb')
            cPickle.dump(self.model.vocab, f, cPickle.HIGHEST_PROTOCOL)
            f.close()
            logging.log(0, 'Done saving w2v minimal model')
        else:
            logging.log(1, 'No model present! Cannot save without model.')
    def load(self, filename=''):
        logging.log(0, 'Loading w2v model...')
        if filename.endswith('.bin'): #Word2Vec format
            self.model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=True)
        else:
            self.model = gensim.models.Word2Vec.load(filename)
        logging.log(0, 'Done loading w2v model.')
    def load_minimal(self, filename, size=400, window=5, sample=1e-4, workers=4, min_count=5, negative=10):
        logging.info('Loading w2v minimal model...')
        f = open(filename + '.1.npy', 'rb')
        self.model = gensim.models.Word2Vec(sentences=None, size=size, window=window, sample=sample, workers=workers, min_count=min_count, negative=negative, hs=0)
        self.model.syn0 = np.load(f)
        f.close()
        f = open(filename + '.2.dump', 'rb')
        self.model.vocab = cPickle.load(f)
        f.close()
        logging.info('Done loading w2v minimal model.')
    def to_lower(self):
        new_vocab = {}
        for word in self.model.vocab:
            lower_word = word.lower()
            if lower_word not in new_vocab:
                new_vocab[lower_word] = self.model.vocab[word]
        self.model.vocab = new_vocab
    def get_vector(self, word=''):
        return self.model[word]
    def exists_word(self, word=''):
        return word in self.model
    def interactive_query(self):
        while True:
            word = raw_input('Type a word: ')
            print self.get_vector(word)
    def interactive_similarity(self):
        while True:
            word1 = raw_input('Type first word: ')
            word2 = raw_input('Type second word: ')
            print self.model.similarity(word1, word2)
    def get_closest_words(self, n=10):
        while True:
            word = raw_input('Type word: ')
            print self.model.most_similar(positive=[word], negative=[], topn=n)


if __name__ == '__main__':
    w = w2v()
    #w.train(folder='../data/w2v/')
    #w.save('w2v_model')
    #w.load('../data/w2v.model')
    #w.save_minimal('../data/minimal')
    w.load_minimal('../data/minimal')
    print w.get_vector('belgium')