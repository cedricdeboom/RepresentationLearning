"""
Stacked Convolutional Autoencoder for character-level Tweet feature extraction with random shifts per epoch (SEMANTIC!)
Based on Theano, (Masci 2011), (dos Santos 2014) and (Djuric 2015)
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 March 16th"

from layers import *
from abstractAE import abstractAE
from reutersProcessor import reutersProcessor
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

class stackedConvAE3(abstractAE):
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
                                       tweet_shape=(reutersProcessor.TWEETDIM, reutersProcessor.NUMCHARS), n_out=10, W=None, dropout=0.1)
        self.layers.append(self.embedding_layer)
        self.params.extend(self.embedding_layer.params)

        #second layer: convolution (no-max pooling)
        self.conv_layer = convEmbeddingCircular(self.numpy_rng, self.theano_rng, input=self.embedding_layer.output.reshape((batch_size, 1, 140, 10)),
                                    tweet_shape=(batch_size,1,140,10), filter_shape=(100,1,5,10), pool_size=(4, 1), dropout=0.15)
        self.layers.append(self.conv_layer)
        self.params.extend(self.conv_layer.params)

        #third layer: MLP
        self.MLP_layer = MLP(self.numpy_rng, self.theano_rng, input=self.conv_layer.output.reshape((batch_size,100*140)), input_shape=(batch_size,100*140),
                             n_out=200, activation=T.nnet.sigmoid, dropout=0.1)
        self.layers.append(self.MLP_layer)
        self.params.extend(self.MLP_layer.params)

        self.output = self.MLP_layer.output


    def finetune_function(self, train_set_x, train_set_shift, batch_size):
        """Extra autoencoding-based finetuning"""
        #SHIFT = INPUT; X = RECONSTRUCTED OUTPUT

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
                                       tweet_shape=(reutersProcessor.TWEETDIM, reutersProcessor.NUMCHARS), n_out=10, W=None, dropout=0.0)
        E.W.set_value(self.embedding_layer.W.get_value(borrow=True))

        C = convEmbeddingCircular(self.numpy_rng, self.theano_rng, input=E.output.reshape((batch_size, 1, 140, 10)),
                                    tweet_shape=(batch_size,1,140,10), filter_shape=(100,1,5,10), pool_size=(4, 1), dropout=0.0)
        C.W.set_value(self.conv_layer.W.get_value(borrow=True))
        C.b.set_value(self.conv_layer.b.get_value(borrow=True))
        C.c.set_value(self.conv_layer.c.get_value(borrow=True))

        M = MLP(self.numpy_rng, self.theano_rng, input=C.output.reshape((batch_size, 100*140)), input_shape=(batch_size, 100*140),
                             n_out=200, activation=T.nnet.sigmoid, dropout=0.0)
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
        return self.MLP_layer.get_hidden_values(output_2.reshape((100,100*140)), batch_size)

    def get_reconstructed_input(self, hidden, batch_size):
        recon_1 = self.MLP_layer.get_reconstructed_input(hidden, batch_size)
        recon_2 = self.conv_layer.get_reconstructed_input(recon_1.reshape((100, 100, 140, 1)), batch_size)
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


    def finetune(self, datafile1=None, datafile2=None, dataset='data/reuters/pairsa_train.npz', batch_size=100, no_of_tweets=1000):
        set1, set2 = reutersProcessor.getData(datafile1, datafile2, dataset, no_of_tweets)
        n_train_batches = set1.shape[0]
        n_train_batches /= batch_size

        train_batch_1 = theano.shared(value=reutersProcessor.transformBatch(set1[0:batch_size, :]), name='train_set_1', borrow=True)
        train_batch_2 = theano.shared(value=reutersProcessor.transformBatch(set2[0:batch_size, :]), name='train_set_2', borrow=True)

        print 'Initializing finetuning function...'
        #dropout_levels = [0.1, 0.15, 0.1]
        dropout_levels = [0.0, 0.0, 0.0]
        learning_rates = [0.00001, 0.00001]
        reg_start = 0.0
        finetune_fn = self.finetune_function(train_batch_2, train_batch_1, batch_size)
        for i in xrange(len(self.layers)):
            self.layers[i].dropout = dropout_levels[i]

        print 'Finetuning model...'
        start_time = time.clock()
        #Train auto-encoder for all layers together
        finetuning_epochs = 5
        for epoch in xrange(finetuning_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                start = batch_index*batch_size
                end = (batch_index+1)*batch_size
                train_batch_1.set_value(reutersProcessor.transformBatch(set1[start:end, :]))
                train_batch_2.set_value(reutersProcessor.transformBatch(set2[start:end, :]))
                cost = finetune_fn(lr=learning_rates[epoch != 0], reg=(epoch>-1) * reg_start)
                c.append(cost)
            print 'Finetuning layers, epoch %d, cost %.5f' % (epoch, numpy.mean(c))
            #self.testSimsData(dumpfile='data/sims.dump', batch_size=2, plotting=epoch+1+20)
            set1, set2 = reutersProcessor.getShiftedData(None, None, no_of_tweets)
            learning_rates[epoch != 0] /= 1.03
            reg_start += (epoch>-1) * 0.001
            reg_start *= (epoch>-1) * 1.2
        self.save_me()

        end_time = time.clock()

        print 'Finetuning done.'
        print 'Finetuning ran for %.2fm.' % ((end_time - start_time) / 60.)

    def test(self, datafile='data/test.txt', dataset='data/test.npz', batch_size=2):
        test_set,_ = reutersProcessor.getData(datafile, dataset, batch_size)
        n_test_batches = test_set.shape[0]
        n_test_batches /= batch_size

        test_set = T._shared(value=reutersProcessor.transformBatch(test_set), name='test_set', borrow=True)

        print 'Initializing test function...'
        test_fn = self.test_function(test_set, batch_size)

        print 'Computing output values...'
        output = test_fn(index = 0)

        return output

    def testSimsData(self, dumpfile='data/sims.dump', batch_size=2, plotting=0):
        (tweets, sims) = reutersProcessor.readSimTweets(dumpfile)

        test_set = T._shared(value=reutersProcessor.transformBatch(tweets[0]), name='test_set', borrow=True)

        print 'Initializing test function...'
        test_fn = self.test_function(test_set, batch_size)

        outputs = []
        h = numpy.array([])
        for t in tweets:
            test_set.set_value(reutersProcessor.transformBatch(t))
            output = test_fn(index = 0)
            if(plotting != 0):
                h = numpy.append(h, output)
            output[0] = [round(i) for i in output[0]]
            output[1] = [round(i) for i in output[1]]
            outputs.append(1-scipy.spatial.distance.hamming(output[0], output[1]))

        if(plotting != 0):
            plt.clf()
            _, _, _ = plt.hist(h, 100, normed=1, facecolor='green', alpha=0.75)
            plt.savefig('plot_' + str(plotting) + '.png')
        (r, p) = util.pearsonCorrelation(sims, outputs)
        print r


if __name__ == '__main__':
    theano.config.compute_test_value = 'off'
    SCAE = stackedConvAE3()
    print 'Initializing model...'
    SCAE.init_model(batch_size=100)
    #SCAE.layers[0].load_me(filename='layer_0.sav')
    #SCAE.layers[1].load_me(filename='layer_1.sav')
    #SCAE.load_me()
    #SCAE.testSimsData(dumpfile='data/sims.dump', batch_size=2)

    #SCAE.pretrain(batch_size=100, no_of_tweets=200000)
    SCAE.finetune(batch_size=100, no_of_tweets=240000)
    #output = SCAE.testSimsData(dumpfile='data/sims.dump', batch_size=2, plotting=0)

    #print output

    #print (1-scipy.spatial.distance.cosine(output[0], output[1]))