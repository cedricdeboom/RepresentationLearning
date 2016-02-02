"""
NN trainer for word embeddings
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.2"
__date__    = "2015 April 28th"

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


NO_DATA = 3000000
BATCH_SIZE = 100
EMBEDDING_DIM = 400
WORDS = 20
INPUT_SHAPE = (EMBEDDING_DIM, WORDS)
OUTPUT_SHAPE = (EMBEDDING_DIM, 1)

BATCHES = NO_DATA / BATCH_SIZE

"""
NN2 trains a small neural network in which we use an MLP to combine word vectors, all through an MLPLayer.
Note we only use an MLP with weight matrix shape (num_words, 1).
"""

class NN2():
    def __init__(self):
        self.numpy_rng = numpy.random.RandomState(89677)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        self.x1 = T.tensor3('x1')
        self.x2 = T.tensor3('x2')
        self.y = T.vector('y') #0 or 1
        self.z = T.vector('z') #-1 or 1

        self.model1 = NN_layers.MLPLayer(self.numpy_rng, self.theano_rng, self.x1, INPUT_SHAPE, n_out=OUTPUT_SHAPE[1], batch_size=BATCH_SIZE, W=None, b=None)
        self.model2 = NN_layers.MLPLayer(self.numpy_rng, self.theano_rng, self.x2, INPUT_SHAPE, n_out=OUTPUT_SHAPE[1], batch_size=BATCH_SIZE, W=self.model1.W, b=self.model1.b)

        self.params = []
        self.params.extend(self.model1.params)

    def run(self, epochs=1, learning_rate=1.5, regularization=0.0, momentum=0.1):
        processor = NN_process.unsortedPairProcessor('../data/pairs/sets/enwiki_pairs_20-train.txt', '../data/pairs/sets/enwiki_no_pairs_20-train.txt',
                                  '../data/model/docfreq.npy', '../data/model/minimal', WORDS, EMBEDDING_DIM, BATCH_SIZE)
        train_x1 = theano.shared(value=processor.x1, name='train_x1', borrow=False)
        train_x2 = theano.shared(value=processor.x2, name='train_x2', borrow=False)
        train_y = theano.shared(value=processor.y, name='train_y', borrow=False)
        train_z = theano.shared(value=processor.z, name='train_z', borrow=False)

        print 'Initializing train function...'
        train = self.train_function_momentum(train_x1, train_x2, train_y, train_z)

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
            train_y.set_value(processor.y, borrow=False)
            train_z.set_value(processor.z, borrow=False)

            processor.lock.acquire()
            processor.cont = True
            processor.ready = False
            processor.lock.notifyAll()
            processor.lock.release()

            for b in xrange(BATCHES):
                #c = []
                cost = train(lr=learning_rate, reg=regularization, mom=momentum)
                #c.append(cost)

                print 'Training, batch %d, cost %.5f' % (b, cost)
                print numpy.transpose(self.model1.W.get_value())

                processor.lock.acquire()
                while not processor.ready:
                    processor.lock.wait()
                processor.lock.release()

                train_x1.set_value(processor.x1, borrow=False)
                train_x2.set_value(processor.x2, borrow=False)
                train_y.set_value(processor.y, borrow=False)
                train_z.set_value(processor.z, borrow=False)

                processor.lock.acquire()
                processor.cont = True
                processor.ready = False
                if b == BATCHES-1 and e == epochs-1:
                    processor.stop = True
                processor.lock.notifyAll()
                processor.lock.release()

            #print 'Training, epoch %d, cost %.5f' % (e, numpy.mean(c))

        self.save_me('run2.npy')

    def save_me(self, filename=None):
        self.model1.save_me(filename)

    def load_me(self, filename=None):
        self.model1.load_me(filename)
        self.model2.W = self.model1.W
        self.model2.b = self.model1.b

    def train_function(self, x1, x2, y, z):
        """Train model"""

        learning_rate = T.scalar('lr')  # learning rate to use
        regularization = T.scalar('reg')  # regularization to use

        cost, updates = self.get_cost_updates(learning_rate, regularization)

        train_fn = theano.function(
            inputs=[
                theano.Param(learning_rate, default=0.0001),
                theano.Param(regularization, default=0.0)
            ],
            outputs=cost,
            updates=updates,
            givens={
                self.x1: x1,
                self.x2: x2,
                self.y: y,
                self.z: z
            },
            name='train'
        )

        return train_fn

    def train_function_momentum(self, x1, x2, y, z):
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
                self.y: y,
                self.z: z
            },
            name='train_momentum'
        )

        return train_fn


    def get_cost_updates(self, learning_rate, regularization):
        """Calculate updates of params based on gradient descent"""

        cost0 = self.calculate_cost()
        cost = cost0 + regularization * self.calculate_regularization()

        # compute the gradients of the cost with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost0, updates)

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
        loss = ((((output1 - output2) ** 2).sum(axis=1))/80.0 - self.y) * (-self.z)
        #distance = 2.0 - (output1 * output2).sum(axis=1) / T.sqrt((output1 * output1).sum(axis=1)) / T.sqrt((output2 * output2).sum(axis=1))
        #loss = (distance - self.y) * (-self.z)
        return T.mean(loss)

    def calculate_regularization(self):
        #return abs(self.model1.W * (self.model1.W < 0.0)).sum() + abs(self.model1.W * (self.model1.W > 1.0)).sum()
        #return (self.model1.W ** 2).sum()
        return 0.0


if __name__ == '__main__':
    n = NN2()
    n.run()