"""
Stacked Convolutional Autoencoder for character-level Tweet feature extraction with random shifts per epoch
Based on Theano, (Masci 2011) and (dos Santos 2014)
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 February 20th"

from layers import *
from abstractAE import abstractAE
from dataProcessor import dataProcessor
import util

import time

import numpy
import scipy
import matplotlib

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

matplotlib.use('Agg')
import matplotlib.pyplot as plt

class stackedConvAE2(abstractAE):
    """Stacked convolutional autoencoder with character embeddings + MLP layer at the end"""

    def __init__(self, numpy_rng=None, theano_rng=None):
        self.layers = []
        self.params = []

        if not numpy_rng:
            self.numpy_rng = numpy.random.RandomState(89677)
        else:
            self.numpy_rng = numpy_rng

        if not theano_rng:
            self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))
        else:
            self.theano_rng = theano_rng

        self.x = T.tensor3('x') #input is represented as minibatches of tweets (-> 3D tensor)
        self.x_shift = T.tensor3('x_shift')

    def save_me(self, filename=None):
        for l in xrange(len(self.layers)):
            self.layers[l].save_me('layer_' + str(l) + '.sav')

    def load_me(self, filename=None):
        for l in xrange(len(self.layers)):
            self.layers[l].load_me('layer_' + str(l) + '.sav')

    def init_model(self, batch_size):
        #first layer: character embeddings
        self.embedding_layer = charEmbedding(self.numpy_rng, self.theano_rng, input=self.x, batch_size=batch_size,
                                       tweet_shape=(dataProcessor.TWEETDIM, dataProcessor.NUMCHARS), n_out=10, W=None, dropout=0.1)
        self.layers.append(self.embedding_layer)
        self.params.extend(self.embedding_layer.params)

        #second layer: convolution + max-pooling
        self.conv_layer = convEmbeddingCircular(self.numpy_rng, self.theano_rng, input=self.embedding_layer.output.reshape((batch_size, 1, 140, 10)),
                                    tweet_shape=(batch_size,1,140,10), filter_shape=(60,1,5,10), pool_size=(4, 1), dropout=0.15)
        self.layers.append(self.conv_layer)
        self.params.extend(self.conv_layer.params)

        #third layer: MLP
        self.MLP_layer = MLP(self.numpy_rng, self.theano_rng, input=self.conv_layer.output.reshape((batch_size,60*140)), input_shape=(batch_size,60*140),
                             n_out=100, activation=T.nnet.sigmoid, dropout=0.1)
        self.layers.append(self.MLP_layer)
        self.params.extend(self.MLP_layer.params)

        self.output = self.MLP_layer.output

    def pretraining_functions(self, train_set_x, batch_size):
        # index to a [mini]batch
        learning_rate = T.scalar('lr')  # learning rate to use
        regularization = T.scalar('reg')  # regularization to use

        pretrain_fns = []
        for layer in self.layers:
            # get the cost and the updates list
            cost, updates = layer.get_cost_updates(learning_rate, batch_size, shift=False, regularization=regularization)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    theano.Param(learning_rate, default=0.001),
                    theano.Param(regularization, default=0.0)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns

    def finetune_function(self, train_set_x, train_set_shift, batch_size):
        """Extra autoencoding-based finetuning"""

        learning_rate = T.scalar('lr')  # learning rate to use
        regularization = T.scalar('reg')  # regularization to use

        cost, updates = self.get_cost_updates(learning_rate, batch_size, shift=True, regularization=regularization)

        train_fn = theano.function(
            inputs=[
                theano.Param(learning_rate, default=0.0001),
                theano.Param(regularization, default=0.0)
            ],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x,
                self.x_shift: train_set_shift,
            },
            name='finetune'
        )

        return train_fn

    def test_function(self, input, batch_size):
        # index to a [mini]batch
        index = T.lscalar('index')

        E = charEmbedding(self.numpy_rng, self.theano_rng, input, batch_size,
                                       tweet_shape=(dataProcessor.TWEETDIM, dataProcessor.NUMCHARS), n_out=10, W=None, dropout=0.0)
        E.W.set_value(self.embedding_layer.W.get_value(borrow=True))

        C = convEmbeddingCircular(self.numpy_rng, self.theano_rng, input=E.output.reshape((batch_size, 1, 140, 10)),
                                    tweet_shape=(batch_size,1,140,10), filter_shape=(60,1,5,10), pool_size=(4, 1), dropout=0.0)
        C.W.set_value(self.conv_layer.W.get_value(borrow=True))
        C.b.set_value(self.conv_layer.b.get_value(borrow=True))
        C.c.set_value(self.conv_layer.c.get_value(borrow=True))

        M = MLP(self.numpy_rng, self.theano_rng, input=C.output.reshape((batch_size, 60*140)), input_shape=(batch_size, 60*140),
                             n_out=100, activation=T.nnet.sigmoid, dropout=0.0)
        M.W.set_value(self.MLP_layer.W.get_value(borrow=True))
        M.b.set_value(self.MLP_layer.b.get_value(borrow=True))
        M.c.set_value(self.MLP_layer.c.get_value(borrow=True))

        test_fn = theano.function(
            inputs=[index],
            outputs=M.output,
            givens={
                E.x: input[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        return test_fn

    def get_hidden_values(self, input, batch_size):
        output_1 = self.embedding_layer.get_hidden_values(input, batch_size)
        output_2 = self.conv_layer.get_hidden_values(output_1.reshape((100, 1, 140, 10)), batch_size)
        return self.MLP_layer.get_hidden_values(output_2.reshape((100,60*140)), batch_size)

    def get_reconstructed_input(self, hidden, batch_size):
        recon_1 = self.MLP_layer.get_reconstructed_input(hidden, batch_size)
        recon_2 = self.conv_layer.get_reconstructed_input(recon_1.reshape((100, 60, 140, 1)), batch_size)
        to_return = self.embedding_layer.get_reconstructed_input(recon_2.reshape((100, 140, 10)), batch_size)
        to_return = T.switch(to_return<0.00001, 0.00001, to_return)
        to_return = T.switch(to_return>0.99999, 0.99999, to_return)
        return to_return

    def calculate_cost(self, x, z):
        """ Cross entropy cost """
        original = T.switch(x<0.00001, 0.00001, x)
        original = T.switch(original>0.99999, 0.99999, original)
        L_temp = T.nnet.binary_crossentropy(z, original).sum(axis=1)
        L = T.sum(L_temp, axis=1)
        return T.mean(L)

    def calculate_regularization(self):
        return self.MLP_layer.calculate_regularization()


    def pretrain(self, datafile='data/london.txt', dataset='data/london.npz', batch_size=100, no_of_tweets=1000):
        train_set, _ = dataProcessor.getData(datafile, dataset, no_of_tweets)
        n_train_batches = train_set.shape[0]
        n_train_batches /= batch_size

        #circular shift of data
        train_set = dataProcessor.getShiftedData(datafile, dataset, no_of_tweets)
        train_batch = theano.shared(value=dataProcessor.transformBatch(train_set[0:batch_size, :]), name='train_batch', borrow=True)

        print 'Initializing pretraining functions...'
        pretraining_fns = self.pretraining_functions(train_batch, batch_size)

        print 'Pretraining model...'
        start_time = time.clock()
        #Train auto-encoder layer by layer
        dropout_levels = [0.0, 0.15, 0.1]
        learning_rate = [0.01, 0.0001, 0.001]
        pretraining_epochs = [4, 4, 20]
        regularization = [0.0, 0.0, 1.0]
        for i in xrange(2,len(self.layers)):
            reg_start = 0.0
            # set dropout levels to 0 for all but current layer
            for j in xrange(len(self.layers)):
                if i == j:
                    self.layers[j].dropout = dropout_levels[j]
                self.layers[j].dropout = 0.0
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs[i]):
                # go through the training set
                c = []
                if(i == 2):
                    util.plotArray(self.MLP_layer.b.get_value(borrow=True).flatten(), 'weights', -1)
                    self.testSimsData(dumpfile='data/sims.dump', batch_size=2, plotting=-1)
                for batch_index in xrange(n_train_batches):
                    start = batch_index*batch_size
                    end = (batch_index+1)*batch_size
                    train_batch.set_value(dataProcessor.transformBatch(train_set[start:end,:]))
                    cost = pretraining_fns[i](
                             lr=learning_rate[i],
                             reg=regularization[i]*reg_start)
                    c.append(cost)
                    if(i == 2):
                        self.testSimsData(dumpfile='data/sims.dump', batch_size=2, plotting=batch_index+1)
                learning_rate[i] /= 1.05
                reg_start += 0.0001
                reg_start *= 1.2
                print 'Pre-training layer %i, epoch %d, cost %.5f' % (i, epoch, numpy.mean(c))
                train_set = dataProcessor.getShiftedData(datafile, dataset, no_of_tweets)
            self.layers[i].save_me('layer_' + str(i) + '.sav')

        end_time = time.clock()

        print 'Pretraining done.'
        print 'Pretraining ran for %.2fm.' % ((end_time - start_time) / 60.)

    def finetune(self, datafile='data/london.txt', dataset='data/london.npz', batch_size=100, no_of_tweets=1000):
        train_set, _ = dataProcessor.getData(datafile, dataset, no_of_tweets)
        n_train_batches = train_set.shape[0]
        n_train_batches /= batch_size

        train_batch = theano.shared(value=dataProcessor.transformBatch(train_set[0:batch_size, :]), name='train_set', borrow=True)
        shifted_set = dataProcessor.getShiftedData(datafile, dataset, no_of_tweets)
        train_batch_shift = theano.shared(value=dataProcessor.transformBatch(shifted_set[0:batch_size, :]), name='train_set_shift', borrow=True)

        print 'Initializing finetuning function...'
        dropout_levels = [0.1, 0.15, 0.1]
        learning_rates = [0.00001, 0.00001]
        reg_start = 0.01
        finetune_fn = self.finetune_function(train_batch, train_batch_shift, batch_size)
        for i in xrange(len(self.layers)):
            self.layers[i].dropout = dropout_levels[i]

        print 'Finetuning model...'
        start_time = time.clock()
        #Train auto-encoder for all layers together
        finetuning_epochs = 20
        for epoch in xrange(finetuning_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                start = batch_index*batch_size
                end = (batch_index+1)*batch_size
                train_batch.set_value(dataProcessor.transformBatch(train_set[start:end, :]))
                train_batch_shift.set_value(dataProcessor.transformBatch(shifted_set[start:end, :]))
                cost = finetune_fn(lr=learning_rates[epoch != 0], reg=(epoch>-1) * reg_start)
                c.append(cost)
            print 'Finetuning layers, epoch %d, cost %.5f' % (epoch, numpy.mean(c))
            self.testSimsData(dumpfile='data/sims.dump', batch_size=2, plotting=epoch+1+20)
            shifted_set = dataProcessor.getShiftedData(datafile, dataset, no_of_tweets)
            learning_rates[epoch != 0] /= 1.05
            reg_start += (epoch>-1) * 0.001
            reg_start *= (epoch>-1) * 1.2
        self.save_me()

        end_time = time.clock()

        print 'Finetuning done.'
        print 'Finetuning ran for %.2fm.' % ((end_time - start_time) / 60.)

    def test(self, datafile='data/test.txt', dataset='data/test.npz', batch_size=2):
        test_set,_ = dataProcessor.getData(datafile, dataset, batch_size)
        n_test_batches = test_set.shape[0]
        n_test_batches /= batch_size

        test_set = T._shared(value=dataProcessor.transformBatch(test_set), name='test_set', borrow=True)

        print 'Initializing test function...'
        test_fn = self.test_function(test_set, batch_size)

        print 'Computing output values...'
        output = test_fn(index = 0)

        return output

    def testSimsData(self, dumpfile='data/sims.dump', batch_size=2, plotting=0):
        (tweets, sims) = dataProcessor.readSimTweets(dumpfile)

        test_set = T._shared(value=dataProcessor.transformBatch(tweets[0]), name='test_set', borrow=True)

        print 'Initializing test function...'
        test_fn = self.test_function(test_set, batch_size)

        outputs = []
        h = numpy.array([])
        for t in tweets:
            test_set.set_value(dataProcessor.transformBatch(t))
            output = test_fn(index = 0)
            if(plotting != 0):
                h = numpy.append(h, output)
            #output[0] = [round(i) for i in output[0]]
            #output[1] = [round(i) for i in output[1]]
            print output
            outputs.append(1-scipy.spatial.distance.hamming(output[0], output[1]))


        if(plotting != 0):
            plt.clf()
            _, _, _ = plt.hist(h, 100, normed=1, facecolor='green', alpha=0.75)
            plt.savefig('plot_' + str(plotting) + '.png')
        (r, p) = util.pearsonCorrelation(sims, outputs)
        print r


if __name__ == '__main__':
    theano.config.compute_test_value = 'off'
    SCAE = stackedConvAE2()
    print 'Initializing model...'
    SCAE.init_model(batch_size=100)
    #SCAE.layers[0].load_me(filename='layer_0.sav')
    #SCAE.layers[1].load_me(filename='layer_1.sav')
    #SCAE.load_me()
    #SCAE.testSimsData(dumpfile='data/sims.dump', batch_size=2)

    #SCAE.pretrain(batch_size=100, no_of_tweets=200000)
    #SCAE.finetune(batch_size=100, no_of_tweets=200000)
    output = SCAE.testSimsData(dumpfile='data/sims.dump', batch_size=2, plotting=0)

    #print output

    #print (1-scipy.spatial.distance.cosine(output[0], output[1]))