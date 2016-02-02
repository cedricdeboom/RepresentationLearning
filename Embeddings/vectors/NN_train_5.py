"""
NN trainer for word embeddings (tweets)
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 August 27th"

import NN_layers
import NN_process


import numpy
import scipy
import matplotlib

from threading import Thread

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

matplotlib.use('Agg')
import matplotlib.pyplot as plt


NO_DATA = 30000
BATCH_SIZE = 100
EMBEDDING_DIM = 400
WORDS = 35  #i.e. max number of words
EPOCHS = 1000
LEARNING_RATE = 0.0000000003
MOMENTUM = 0.9
REGULARIZATION = 0.001

INPUT_SHAPE = (EMBEDDING_DIM, WORDS)
OUTPUT_SHAPE = (EMBEDDING_DIM, 1)
BATCHES = NO_DATA / BATCH_SIZE

"""
NN5 trains a small neural network in which we assign a weight to each vector in the sentence
and take the mean. We learn a quartic weight function through a quarticWeightsDotMeanLayer.
THIS NETWORK IS FOR TWEETS (no wiki articles)!
"""
class NN5():
    def __init__(self):
        self.numpy_rng = numpy.random.RandomState(89677)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        self.x1 = T.tensor3('x1')
        self.x2 = T.tensor3('x2')
        self.l1 = T.vector('l1')
        self.l2 = T.vector('l2')
        self.y = T.vector('y') #0 or 1
        self.z = T.vector('z') #-1 or 1

        self.model1 = NN_layers.quarticWeightsDotMeanLayer(self.numpy_rng, self.theano_rng, self.x1, INPUT_SHAPE,
                                                             self.l1, max_length=35, n_out=OUTPUT_SHAPE[1], batch_size=BATCH_SIZE, W=None)
        self.model2 = NN_layers.quarticWeightsDotMeanLayer(self.numpy_rng, self.theano_rng, self.x2, INPUT_SHAPE,
                                                             self.l2, max_length=35, n_out=OUTPUT_SHAPE[1], batch_size=BATCH_SIZE, W=self.model1.W)

        self.params = []
        self.params.extend(self.model1.params)

    def run(self, epochs=EPOCHS, learning_rate=LEARNING_RATE, regularization=REGULARIZATION, momentum=MOMENTUM):
        processor = NN_process.lengthPairProcessor('../data/tweets/pairs/sets/tweet-pairs-train.txt',
                                             '../data/tweets/pairs/sets/tweet-no-pairs-train.txt',
                                  '../data/wiki/model/docfreq.npy', '../data/wiki/model/minimal', WORDS, EMBEDDING_DIM, BATCH_SIZE)
        train_x1 = theano.shared(value=processor.x1, name='train_x1', borrow=False)
        train_x2 = theano.shared(value=processor.x2, name='train_x2', borrow=False)
        train_l1 = theano.shared(value=processor.l1, name='train_l1', borrow=False)
        train_l2 = theano.shared(value=processor.l2, name='train_l2', borrow=False)
        train_y = theano.shared(value=processor.y, name='train_y', borrow=False)
        train_z = theano.shared(value=processor.z, name='train_z', borrow=False)

        print 'Initializing train function...'
        train = self.train_function_momentum(train_x1, train_x2, train_l1, train_l2, train_y, train_z)

        lin = numpy.linspace(0, float(WORDS) - 1.0, float(WORDS))
        c = []

        t = Thread(target=processor.process)
        t.start()
        for e in xrange(epochs):
            processor.new_epoch()

            processor.lock.acquire()
            while not processor.ready:
                processor.lock.wait()
            processor.lock.release()

            train_x1.set_value(processor.x1, borrow=False)
            train_x2.set_value(processor.x2, borrow=False)
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

                train_x1.set_value(processor.x1, borrow=False)
                train_x2.set_value(processor.x2, borrow=False)
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
                processor.lock.notifyAll()
                processor.lock.release()

            print 'Training, epoch %d, cost %.5f' % (e, numpy.mean(c))
            we = self.model1.W.get_value()
            print we
            f = we[0] * (lin ** 4) + we[1] * (lin ** 3) + we[2] * (lin ** 2) + we[3] * lin + 1.0
            print f

        self.save_me('run5.npy')

    def save_me(self, filename=None):
        self.model1.save_me(filename)

    def load_me(self, filename=None):
        self.model1.load_me(filename)
        self.model2.W = self.model1.W

    def train_function_momentum(self, x1, x2, l1, l2, y, z):
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
                self.l1: l1,
                self.l2: l2,
                self.y: y,
                self.z: z
            },
            name='train_momentum'
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
        weights = (self.model1.factors_30 ** 2).sum()  #--> EUCLIDEAN LOSS
        loss = ((((output1 - output2) ** 2).sum(axis=1) / weights) - self.y) * (-self.z)  #--> EUCLIDEAN LOSS
        #distance = (1.0 - (output1 * output2).sum(axis=1) / T.sqrt((output1 * output1).sum(axis=1)) / T.sqrt((output2 * output2).sum(axis=1))) / 2.0  #--> COSINE LOSS
        #loss = (distance - self.y) * (-self.z)  #--> COSINE LOSS
        return T.mean(loss)

    def calculate_regularization(self):
        return (abs(self.model1.factors_30[-1] - 0.0))**2 + (abs((self.model1.W[0]*157216 + self.model1.W[1]*3468 + self.model1.W[2]*68 + self.model1.W[3]) - 0.0))**2
        #return 0.0


if __name__ == '__main__':
    n = NN5()
    n.run()