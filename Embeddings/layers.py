"""
Stacked Convolutional Autoencoder for character-level Tweet feature extraction
Based on Theano, (Masci 2011) and (dos Santos 2014)
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.2"
__date__    = "2015 February 11th"


import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from abstractAE import abstractAE

BETA = 5

"""rectified linear unit (ReLU)"""
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

"""Continuous Log-Sigmoid Function"""
def sigmoid(x):
    y = 1 / (1 + T.exp(-BETA * x))
    return y

class charEmbedding(abstractAE):
    """Calculator of distributed character embeddings"""
    
    def __init__(self, numpy_rng, theano_rng, input, batch_size=100, tweet_shape=(140,42), n_out=10, W=None, dropout=0.0):
        self.n_in = tweet_shape[1]
        self.n_out = n_out
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.dropout = dropout
        
        if W is None:
            W_values = numpy.asarray(
               numpy_rng.uniform(
                    low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
                    high=numpy.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)
        
        self.W = W
        self.x = input           #minibatch of tweets, represented as 3D tensor
        self.index = T.lscalar() #input to the layer
        self.output = self.get_corrupted_input(
            T.dot(self.x, self.W), corruption_level=self.dropout)
        self.params = [self.W]

    def save_me(self, filename=None):
        f = open(filename, 'wb')
        numpy.savez_compressed(f, self.W.get_value(borrow=True))
        f.close()

    def load_me(self, filename=None):
        f = open(filename, 'rb')
        dictionary = numpy.load(f)
        self.W.set_value(dictionary['arr_0'])
        f.close()
    
    def get_hidden_values(self, input, batch_size):
        return self.get_corrupted_input(
            T.dot(input, self.W), corruption_level=self.dropout)

    def get_reconstructed_input(self, hidden, batch_size):
        to_return = T.dot(hidden, self.W.T)
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
        return 0.0


class maxPool(abstractAE):
    """Max-pool layer for pooling convolution layer output"""
    """Helper class for convolutional layer"""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, pool_size=(4, 1)):
        self.x = input
        self.pool_size = pool_size
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        # downsample each feature map individually, using max-pooling
        pooled_out = downsample.max_pool_2d(
            input=self.x,
            ds=self.pool_size,
            ignore_border=True
        )

        zero_values = numpy.zeros(input_shape, dtype=theano.config.floatX)
        self.zeros = theano.shared(value=zero_values, borrow=True)
        self.output = pooled_out

    def save_me(self, filename=None):
        pass

    def load_me(self, filename=None):
        pass

    def get_hidden_values(self, input, batch_size):
        return downsample.max_pool_2d(
            input=input,
            ds=self.pool_size,
            ignore_border=True
        )

    def get_reconstructed_input(self, hidden, batch_size):
        #add zeros in between hidden values to obtain tensor with same dims as input to the max-pool layer
        return T.set_subtensor(self.zeros[:, :, ::self.pool_size[0], :], hidden)

    def calculate_cost(self, x, z):
        return 0.0

    def calculate_regularization(self):
        return 0.0


class convEmbedding(abstractAE):
    """Convolutional layer transforming character embeddings + max-pooling"""
    
    def __init__(self, numpy_rng, theano_rng, input, tweet_shape=(100,1,140,10), filter_shape=(40,1,5,10), pool_size=(4, 1), maxpool=False, dropout=0.0):
        """
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (num kernels, num input feature maps,
                              filter height, filter width)

        :type tweet_shape: tuple or list of length 4
        :param tweet_shape: (batch size, num input feature maps,
                             image height, image width)
        """
        assert tweet_shape[1] == filter_shape[1]
        self.x = input
        self.tweet_shape = tweet_shape
        self.filter_shape = filter_shape
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.maxpool = maxpool
        self.dropout = dropout
        
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:])) / numpy.prod(pool_size)
        
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        #bias values for every feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        #bias values for every input feature map (used for decoding)
        c_values = numpy.zeros((filter_shape[1],), dtype=theano.config.floatX)
        self.c = theano.shared(value=c_values, borrow=True)
        
        conv_out = conv.conv2d(
            input=self.x,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=tweet_shape
        )

        output_conv = T.nnet.sigmoid(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        if self.maxpool:
            output_shape = (tweet_shape[0], filter_shape[0], tweet_shape[2]-filter_shape[2]+1, tweet_shape[3]-filter_shape[3]+1)
            self.pool = maxPool(self.numpy_rng, self.theano_rng, output_conv, input_shape=output_shape, pool_size=pool_size)
            self.output = self.get_corrupted_input(
                self.pool.output, corruption_level=self.dropout)
        else:
            self.output = self.get_corrupted_input(
                output_conv, corruption_level=self.dropout)
        self.params = [self.W, self.b, self.c]

    def save_me(self, filename=None):
        f = open(filename, 'wb')
        numpy.savez_compressed(f, self.W.get_value(borrow=True), self.b.get_value(borrow=True), self.c.get_value(borrow=True))
        f.close()

    def load_me(self, filename=None):
        f = open(filename, 'rb')
        dictionary = numpy.load(f)
        self.W.set_value(dictionary['arr_0'])
        self.b.set_value(dictionary['arr_1'])
        self.c.set_value(dictionary['arr_2'])
        f.close()
    
    def get_hidden_values(self, input, batch_size):
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.tweet_shape
        )
        output_conv = T.nnet.sigmoid(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        if self.maxpool:
            return self.get_corrupted_input(
                self.pool.get_hidden_values(output_conv, batch_size), corruption_level=self.dropout)
        return self.get_corrupted_input(
            output_conv, corruption_level=self.dropout)

    def get_reconstructed_input(self, hidden, batch_size):
        if self.maxpool:
            pool_hidden = self.pool.get_reconstructed_input(hidden, batch_size)
        else:
            pool_hidden = hidden
        image_shape_flip = (self.tweet_shape[0], self.filter_shape[0],
                          self.tweet_shape[2]-self.filter_shape[2]+1,
                          self.tweet_shape[3]-self.filter_shape[3]+1)
        filter_shape_flip = (self.tweet_shape[1], self.filter_shape[0],
                          self.filter_shape[2], self.filter_shape[3])
        W_flip = self.W[:, :, ::-1, ::-1].reshape(filter_shape_flip) #flip columns and rows of the filters
        conv_out = conv.conv2d( #calculate reverse convolution
            input=pool_hidden,
            filters=W_flip,
            image_shape=image_shape_flip,
            filter_shape=filter_shape_flip,
            border_mode='full' #use full border to increase dims
        )
        #following line is not necessary: already provided by the convolution operation
        #conv_out = T.sum(conv_out, axis=1, keepdims=True) #sum along axis of source feature maps
        return T.nnet.sigmoid(conv_out + self.c.dimshuffle('x', 0, 'x', 'x')) #add bias to each input channel
        
    def calculate_cost(self, x, z):
        """ MSE cost """
        L_temp = ((x - z) ** 2).sum(axis=2).sum(axis=2).sum(axis=1)
        return T.mean(L_temp)

    def calculate_regularization(self):
        return 0.0


class convEmbeddingCircular(abstractAE):
    """Convolutional layer transforming character embeddings + max-pooling (circular)"""

    def __init__(self, numpy_rng, theano_rng, input, tweet_shape=(100,1,140,10), filter_shape=(40,1,5,10), pool_size=(4, 1), maxpool=False, dropout=0.0):
        """
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (num kernels, num input feature maps,
                              filter height, filter width)

        :type tweet_shape: tuple or list of length 4
        :param tweet_shape: (batch size, num input feature maps,
                             image height, image width)
        """
        assert tweet_shape[1] == filter_shape[1]
        self.x = input
        self.original_shape = tweet_shape
        self.tweet_shape = (tweet_shape[0], tweet_shape[1], tweet_shape[2] + filter_shape[2] - 1, tweet_shape[3])
        self.offset = (filter_shape[2] - 1) / 2
        self.filter_shape = filter_shape
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.maxpool = maxpool
        self.dropout = dropout

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:])) / numpy.prod(pool_size)

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        #bias values for every feature map
        b_values = numpy.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        #bias values for every input feature map (used for decoding)
        c_values = numpy.zeros((self.filter_shape[1],), dtype=theano.config.floatX)
        self.c = theano.shared(value=c_values, borrow=True)

        self.convolution_input = theano.shared(value=numpy.zeros((self.tweet_shape), dtype=theano.config.floatX), borrow=True)
        #self.reconstructed_output = theano.shared(value=numpy.zeros((self.original_shape), dtype=theano.config.floatX), borrow=True)

        self.convolution_input = T.set_subtensor(self.convolution_input[:,:,self.offset:-self.offset,:], self.x)
        self.convolution_input = T.set_subtensor(self.convolution_input[:,:,:self.offset,:], self.x[:,:,-self.offset:,:])
        self.convolution_input = T.set_subtensor(self.convolution_input[:,:,-self.offset:,:], self.x[:,:,:self.offset,:])

        conv_out = conv.conv2d(
            input=self.convolution_input,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.tweet_shape
        )

        output_conv = T.nnet.sigmoid(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        if self.maxpool:
            output_shape = (tweet_shape[0], filter_shape[0], tweet_shape[2]-filter_shape[2]+1, tweet_shape[3]-filter_shape[3]+1)
            self.pool = maxPool(self.numpy_rng, self.theano_rng, output_conv, input_shape=output_shape, pool_size=pool_size)
            self.output = self.get_corrupted_input(
                self.pool.output, corruption_level=self.dropout)
        else:
            self.output = self.get_corrupted_input(
                output_conv, corruption_level=self.dropout)
        self.params = [self.W, self.b, self.c]

    def save_me(self, filename=None):
        f = open(filename, 'wb')
        numpy.savez_compressed(f, self.W.get_value(borrow=True), self.b.get_value(borrow=True), self.c.get_value(borrow=True))
        f.close()

    def load_me(self, filename=None):
        f = open(filename, 'rb')
        dictionary = numpy.load(f)
        self.W.set_value(dictionary['arr_0'])
        self.b.set_value(dictionary['arr_1'])
        self.c.set_value(dictionary['arr_2'])
        f.close()

    def get_hidden_values(self, input, batch_size):
        self.convolution_input = T.set_subtensor(self.convolution_input[:,:,self.offset:-self.offset,:], input)
        self.convolution_input = T.set_subtensor(self.convolution_input[:,:,:self.offset,:], input[:,:,-self.offset:,:])
        self.convolution_input = T.set_subtensor(self.convolution_input[:,:,-self.offset:,:], input[:,:,:self.offset,:])

        conv_out = conv.conv2d(
            input=self.convolution_input,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.tweet_shape
        )

        output_conv = T.nnet.sigmoid(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        if self.maxpool:
            return self.get_corrupted_input(
                self.pool.get_hidden_values(output_conv, batch_size), corruption_level=self.dropout)
        return self.get_corrupted_input(
            output_conv, corruption_level=self.dropout)

    def get_reconstructed_input(self, hidden, batch_size):
        if self.maxpool:
            pool_hidden = self.pool.get_reconstructed_input(hidden, batch_size)
        else:
            pool_hidden = hidden
        image_shape_flip = (self.tweet_shape[0], self.filter_shape[0],
                          self.tweet_shape[2]-self.filter_shape[2]+1,
                          self.tweet_shape[3]-self.filter_shape[3]+1)
        filter_shape_flip = (self.tweet_shape[1], self.filter_shape[0],
                          self.filter_shape[2], self.filter_shape[3])
        W_flip = self.W[:, :, ::-1, ::-1].reshape(filter_shape_flip) #flip columns and rows of the filters
        conv_out = conv.conv2d( #calculate reverse convolution
            input=pool_hidden,
            filters=W_flip,
            image_shape=image_shape_flip,
            filter_shape=filter_shape_flip,
            border_mode='full' #use full border to increase dims
        )

        conv_out = conv_out[:,:,self.offset:-self.offset,:]

        return T.nnet.sigmoid(conv_out + self.c.dimshuffle('x', 0, 'x', 'x')) #add bias to each input channel

    def calculate_cost(self, x, z):
        """ MSE cost """
        L_temp = ((x - z) ** 2).sum(axis=2).sum(axis=2).sum(axis=1)
        return T.mean(L_temp)

    def calculate_regularization(self):
        return 0.0


class MLP(abstractAE):
    """MLP layer for final non-linear transformation"""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, n_out=25, activation=T.nnet.sigmoid, dropout=0.0):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input
        self.activation = activation
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.dropout = dropout

        W_values = numpy.asarray(
                self.numpy_rng.uniform(
                    low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
                    high=numpy.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_out)
                ),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        c_values = numpy.zeros((self.n_in,), dtype=theano.config.floatX)
        self.c = theano.shared(value=c_values, name='c', borrow=True)

        self.output = self.get_corrupted_input(
            self.activation(T.dot(self.x, self.W) + self.b), corruption_level=self.dropout)
        self.params = [self.W, self.b, self.c]

    def save_me(self, filename=None):
        f = open(filename, 'wb')
        numpy.savez_compressed(f, self.W.get_value(borrow=True), self.b.get_value(borrow=True), self.c.get_value(borrow=True))
        f.close()

    def load_me(self, filename=None):
        f = open(filename, 'rb')
        dictionary = numpy.load(f)
        self.W.set_value(dictionary['arr_0'])
        self.b.set_value(dictionary['arr_1'])
        self.c.set_value(dictionary['arr_2'])
        f.close()

    def get_hidden_values(self, input, batch_size):
        return self.get_corrupted_input(
            self.activation(T.dot(input, self.W) + self.b), corruption_level=self.dropout)

    def get_reconstructed_input(self, hidden, batch_size):
        return self.activation(T.dot(hidden, self.W.T) + self.c)

    def calculate_cost(self, x, z):
        """ MSE cost """
        L_temp = ((x - z) ** 2).sum(axis=1)
        return T.mean(L_temp)

    def calculate_regularization(self):
        return 0.0