'''
Abstract class for an Autoencoder
'''
__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.2"
__date__    = "2015 February 11th"

import abc
import numpy

import theano
import theano.tensor as T

class abstractAE(object):
    """Abstract autoencoder class"""
    __metaclass__  = abc.ABCMeta
    
    def get_corrupted_input(self, input, corruption_level):
        """Add noise to input"""
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    @abc.abstractmethod
    def save_me(self, filename=None):
        """Abstract"""
        raise NotImplementedError("Method is not implemented in this class.")

    @abc.abstractmethod
    def load_me(self, filename=None):
        """Abstract"""
        raise NotImplementedError("Method is not implemented in this class.")
    
    @abc.abstractmethod
    def get_hidden_values(self, input, batch_size):
        """Abstract"""
        raise NotImplementedError("Method is not implemented in this class.")
    
    @abc.abstractmethod
    def get_reconstructed_input(self, hidden, batch_size):
        """Abstract"""
        raise NotImplementedError("Method is not implemented in this class.")
    
    @abc.abstractmethod
    def calculate_cost(self, z):
        """Abstract"""
        raise NotImplementedError("Method is not implemented in this class.")

    @abc.abstractmethod
    def calculate_regularization(self):
        """Abstract"""
        raise NotImplementedError("Method is not implemented in this class.")
    
    def get_cost_updates(self, learning_rate, batch_size, regularization, shift=False):
        """Calculate updates to autoencoder params based on gradient descent"""
        if not shift:
            tilde_x = self.x
        else:
            tilde_x = self.x_shift
        y = self.get_hidden_values(tilde_x, batch_size)
        z = self.get_reconstructed_input(y, batch_size)

        cost = self.calculate_cost(self.x, z) + regularization * self.calculate_regularization()

        # compute the gradients of the cost with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)