"""
NN trainer for word embeddings (tweets)
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 Nov 4th"

import NN_layers
import NN_process


import numpy
import scipy
import math
import matplotlib
import signal

from threading import Thread

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

matplotlib.use('Agg')
import matplotlib.pyplot as plt


NO_DATA = 40000
BATCH_SIZE = 100
EMBEDDING_DIM = 400
WORDS = 35  #i.e. max number of words
EPOCHS = 500
LEARNING_RATE = 0.001
MOMENTUM = 0.0
REGULARIZATION = 0.0

INPUT_SHAPE = (EMBEDDING_DIM, WORDS)
OUTPUT_SHAPE = (EMBEDDING_DIM, 1)
BATCHES = NO_DATA / BATCH_SIZE

"""
NN9 trains a small neural network in which we assign a linear weight to each vector in the sentence.
We learn the weights directly and linearly.
THIS NETWORK IS FOR TWEETS (no wiki articles)!
"""

class NN9():
    def __init__(self):
        self.numpy_rng = numpy.random.RandomState(89677)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        self.x1 = T.tensor3('x1')
        self.x2 = T.tensor3('x2')
        self.indices1 = T.matrix('i1')
        self.indices2 = T.matrix('i2')
        self.l1 = T.vector('l1')
        self.l2 = T.vector('l2')
        self.y = T.vector('y') #0 or 1
        self.z = T.vector('z') #-1 or 1

        self.model1 = NN_layers.linDotMeanLayer(self.numpy_rng, self.theano_rng, self.x1, INPUT_SHAPE,
                                                             self.indices1, self.l1, max_length=35, n_out=OUTPUT_SHAPE[1], batch_size=BATCH_SIZE, W=None)
        self.model2 = NN_layers.linDotMeanLayer(self.numpy_rng, self.theano_rng, self.x2, INPUT_SHAPE,
                                                             self.indices2, self.l2, max_length=35, n_out=OUTPUT_SHAPE[1], batch_size=BATCH_SIZE, W=self.model1.W)

        self.params = []
        self.params.extend(self.model1.params)

    def run(self, epochs=EPOCHS, learning_rate=LEARNING_RATE, regularization=REGULARIZATION, momentum=MOMENTUM):
        processor = NN_process.lengthLinTweetPairProcessor('../data/tweets/pairs/sets/tweet-pairs-train.txt',
                                                        '../data/tweets/pairs/sets/tweet-no-pairs-train.txt',
                                  '../data/wiki/model/docfreq.npy', '../data/wiki/model/minimal', WORDS, EMBEDDING_DIM, BATCH_SIZE)
        train_x1 = theano.shared(value=processor.x1, name='train_x1', borrow=False)
        train_x2 = theano.shared(value=processor.x2, name='train_x2', borrow=False)
        train_i1 = theano.shared(value=processor.indices1, name='train_i1', borrow=False)
        train_i2 = theano.shared(value=processor.indices2, name='train_i2', borrow=False)
        train_l1 = theano.shared(value=processor.l1, name='train_l1', borrow=False)
        train_l2 = theano.shared(value=processor.l2, name='train_l2', borrow=False)
        train_y = theano.shared(value=processor.y, name='train_y', borrow=False)
        train_z = theano.shared(value=processor.z, name='train_z', borrow=False)

        print 'Initializing train function...'
        train = self.train_function_momentum(train_x1, train_x2, train_i1, train_i2, train_l1, train_l2, train_y, train_z)

        c = []

        t = Thread(target=processor.process)
        t.daemon = True
        t.start()

        def signal_handler(signal, frame):
            import os
            os._exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        best_cost = float('inf')
        best_weights = None

        for e in xrange(epochs):
            processor.new_epoch()

            processor.lock.acquire()
            while not processor.ready:
                processor.lock.wait()
            processor.lock.release()

            train_x1.set_value(processor.x1, borrow=False)
            train_x2.set_value(processor.x2, borrow=False)
            train_i1.set_value(processor.indices1, borrow=False)
            train_i2.set_value(processor.indices2, borrow=False)
            train_l1.set_value(processor.l1, borrow=False)
            train_l2.set_value(processor.l2, borrow=False)
            train_y.set_value(processor.y, borrow=False)
            train_z.set_value(processor.z, borrow=False)

            processor.lock.acquire()
            processor.cont = True
            processor.ready = False
            processor.lock.notifyAll()
            processor.lock.release()

            for b in xrange(BATCHES):
                c = []
                cost = train(lr=learning_rate, reg=regularization, mom=momentum)
                c.append(cost)

                processor.lock.acquire()
                while not processor.ready:
                    processor.lock.wait()
                processor.lock.release()

                # print 'Training, batch %d (from %d), cost %.5f' % (b, BATCHES, cost)
                # we = self.model1.W.get_value()
                # print we

                train_x1.set_value(processor.x1, borrow=False)
                train_x2.set_value(processor.x2, borrow=False)
                train_i1.set_value(processor.indices1, borrow=False)
                train_i2.set_value(processor.indices2, borrow=False)
                train_l1.set_value(processor.l1, borrow=False)
                train_l2.set_value(processor.l2, borrow=False)
                train_y.set_value(processor.y, borrow=False)
                train_z.set_value(processor.z, borrow=False)

                processor.lock.acquire()
                if b < BATCHES-2:
                    processor.cont = True
                    processor.ready = False
                if b == BATCHES-1 and e == epochs-1:
                    processor.stop = True
                    processor.cont = True
                processor.lock.notifyAll()
                processor.lock.release()

            print 'Training, epoch %d, cost %.5f' % (e, numpy.mean(c))
            we = self.model1.W.get_value()
            print repr(we)

            # if numpy.mean(c) < best_cost - 0.0001:
            #     best_cost = numpy.mean(c)
            #     best_weights = we
            # else:
            #     processor.lock.acquire()
            #     processor.stop = True
            #     processor.cont = True
            #     processor.lock.notifyAll()
            #     processor.lock.release()
            #     break

        t.join()
        return best_weights

    def save_me(self, filename=None):
        self.model1.save_me(filename)

    def load_me(self, filename=None):
        self.model1.load_me(filename)
        self.model2.W = self.model1.W

    def train_function_momentum(self, x1, x2, i1, i2, l1, l2, y, z):
        """Train model with momentum"""

        learning_rate = T.scalar('lr')  # learning rate to use
        regularization = T.scalar('reg')  # regularization to use
        momentum = T.scalar('mom')  # momentum to use

        cost, updates = self.get_cost_updates_momentum(learning_rate, regularization, momentum)

        train_fn = theano.function(
            inputs=[
                theano.Param(learning_rate, default=0.1),
                theano.Param(regularization, default=0.0),
                theano.Param(momentum, default=0.9)
            ],
            outputs=cost,
            updates=updates,
            givens={
                self.x1: x1,
                self.x2: x2,
                self.indices1: i1,
                self.indices2: i2,
                self.l1: l1,
                self.l2: l2,
                self.y: y,
                self.z: z
            },
            name='train_momentum',
            on_unused_input='warn'
        )

        return train_fn

    def get_cost_updates_momentum(self, learning_rate, regularization, momentum):
        """Calculate updates of params based on momentum gradient descent"""

        cost0 = self.calculate_cost()
        cost = cost0 + regularization * self.calculate_regularization()
        gparams = T.grad(cost, self.params)

        updates = []
        for p, g in zip(self.params, gparams):
            mparam_i = theano.shared(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v = momentum * mparam_i - learning_rate * g
            updates.append((mparam_i, v))
            updates.append((p, p + v))

        return (cost0, updates)

    def calculate_cost(self):
        output1 = self.model1.output
        output2 = self.model2.output

        #Median loss with cross-entropy
        distances = ((output1 - output2) ** 2).sum(axis=1)
        sorted_distances = T.sort(distances)
        median = (sorted_distances[BATCH_SIZE/2] + sorted_distances[BATCH_SIZE/2 - 1]) / 2.0

        p = distances[0:BATCH_SIZE:2]  #pairs
        q = distances[1:BATCH_SIZE:2]  #non-pairs

        loss = (T.log(1.0 + T.exp(-60.0*(q - median)))).mean() + \
            (T.log(1.0 + T.exp(-60.0*(median - p)))).mean()          #cross-entropy
        return loss

    def calculate_regularization(self):
        return (self.model1.W ** 2).sum()


if __name__ == '__main__':
    n = NN9()
    n.run()