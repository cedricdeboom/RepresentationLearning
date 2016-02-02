"""
NN Layers for word embedding function training
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.1"
__date__    = "2015 April 22nd"


import numpy

import theano
import theano.tensor as T



class dotMeanLayer():
    """Layer calculating dot product of input with weights and taking the mean"""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, n_out=1, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        if W is None:
            W_values = numpy.asarray(
                    self.numpy_rng.uniform(
                        low=0.5,
                        high=0.5,
                        size=(self.n_in)
                    ),
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.output = T.mean(self.x * self.W, axis=2)

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
        return T.mean(self.x * self.W, axis=2)


class doubleDotMeanLayer():
    """Layer calculating dot product of input with weights and second input and taking the mean"""

    def __init__(self, numpy_rng, theano_rng, input, input2, input_shape, n_out=1, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.x_bis = input2 #2D tensor
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        if W is None:
            W_values = numpy.asarray(
                    self.numpy_rng.uniform(
                        low=0.5,
                        high=0.5,
                        size=(self.n_in)
                    ),
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.output = T.mean(self.x * self.x_bis.dimshuffle(0, 'x', 1) * self.W, axis=2)

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
        return T.mean(self.x * self.x_bis.dimshuffle(0, 'x', 1) * self.W, axis=2)


class quadraticWeightsDotMeanLayer():
    """Layer calculating dot product of input with weights and taking the mean.
       Weights are calculated through a quadratic function, of which the coefficients
       are begin learned (intercept = 1.0)."""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, length, max_length=30, n_out=1, batch_size=100, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.length = length #1D tensor
        self.max_length = float(max_length)
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.batch_size = batch_size

        if W is None:
            W_values = numpy.asarray(
                    #[0.00102632, -0.06468767],
                    [0.0012 * (30.0/self.max_length)**2, -0.067 * (30.0/self.max_length)],
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.A = theano.shared(value=numpy.tile((numpy.asarray([numpy.linspace(0, self.max_length-1.0, self.max_length)])).transpose(), (1, self.batch_size)),
                               name='A', borrow=True)
        self.factors = self.W[0] * ((self.A * self.max_length / (self.length+1.0)) ** 2) + self.W[1] * (self.A * self.max_length / (self.length+1.0)) + 1.0
        self.factors_norm = self.factors / self.factors[0]
        self.output = T.sum(self.x * T.transpose(self.factors_norm).dimshuffle(0, 'x', 1), axis=2) / (self.length.dimshuffle(0, 'x') + 1.0)

        self.params = [self.W]
        self.lin = theano.shared(value=numpy.linspace(0, self.max_length - 1.0, self.max_length), name='lin', borrow=True)
        self.factors_30 = (self.W[0] * (self.lin ** 2)) + (self.W[1] * self.lin) + 1.0
        #self.factors_30_norm = self.factors_30 / self.factors_30[0]

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
        self.factors = self.W[0] * ((self.A * self.max_length / self.length) ** 2) + self.W[1] * (self.A * self.max_length / self.length) + 1.0
        return T.sum(self.x * T.transpose(self.factors_norm).dimshuffle(0, 'x', 1), axis=2) / (self.length.dimshuffle(0, 'x') + 1.0)


class quarticWeightsDotMeanLayer():
    """Layer calculating dot product of input with weights and taking the mean.
       Weights are calculated through a quartic function, of which the coefficients
       are begin learned (intercept = 1.0)."""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, length, max_length=30, n_out=1, batch_size=100, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.length = length #1D tensor
        self.max_length = float(max_length)
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.batch_size = batch_size

        if W is None:
            W_values = numpy.asarray(
                    #[0.00102632, -0.06468767],
                    #[0.000002577 * (30.0/self.max_length)**4, -0.0001461 * (30.0/self.max_length)**3,
                    # 0.00377 * (30.0/self.max_length)**2, -0.08198 * (30.0/self.max_length)],
                    #[1.64569337E-6, -1.00672289E-4, 2.76979594E-3, -7.02685714E-2],
                    [9.55526843e-07,  -6.23865853e-05,   2.29907897e-03,  -7.28000030e-02],
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.A = theano.shared(value=numpy.tile((numpy.asarray([numpy.linspace(0, self.max_length-1.0, self.max_length)])).transpose(), (1, self.batch_size)),
                               name='A', borrow=True)
        self.factors = self.W[0] * ((self.A * self.max_length / (self.length+1.0)) ** 4) + self.W[1] * ((self.A * self.max_length / (self.length+1.0)) ** 3) + \
                       self.W[2] * ((self.A * self.max_length / (self.length+1.0)) ** 2) + self.W[3] * (self.A * self.max_length / (self.length+1.0)) + 1.0
        self.factors_norm = self.factors / self.factors[0]
        self.output = T.sum(self.x * T.transpose(self.factors_norm).dimshuffle(0, 'x', 1), axis=2) / (self.length.dimshuffle(0, 'x') + 1.0)

        self.params = [self.W]
        self.lin = theano.shared(value=numpy.linspace(0, self.max_length - 1.0, self.max_length), name='lin', borrow=True)
        self.factors_30 = (self.W[0] * (self.lin ** 4)) + (self.W[1] * (self.lin ** 3)) + (self.W[2] * (self.lin ** 2)) + (self.W[3] * self.lin) + 1.0
        #self.factors_30_norm = self.factors_30 / self.factors_30[0]

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
        self.factors = self.W[0] * ((self.A * self.max_length / (self.length+1.0)) ** 4) + self.W[1] * ((self.A * self.max_length / (self.length+1.0)) ** 3) + \
                       self.W[2] * ((self.A * self.max_length / (self.length+1.0)) ** 2) + self.W[3] * (self.A * self.max_length / (self.length+1.0)) + 1.0
        self.factors_norm = self.factors / self.factors[0]
        return T.sum(self.x * T.transpose(self.factors_norm).dimshuffle(0, 'x', 1), axis=2) / (self.length.dimshuffle(0, 'x') + 1.0)


class exponentialWeightsDotMeanLayer():
    """Layer calculating dot product of input with weights and taking the mean.
       Weights are calculated through an exponential function, of which the coefficients
       are begin learned (intercept = 1.0)."""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, length, max_length=30, n_out=1, batch_size=100, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.length = length #1D tensor
        self.max_length = float(max_length)
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.batch_size = batch_size

        if W is None:
            W_values = numpy.asarray([-9.46516705], # exp(a + b*x) + (1 - exp(a)); b = ln(1 - exp(-a))/(max_length-1); a = sigmoid(param)
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.A = theano.shared(value=numpy.tile((numpy.asarray([numpy.linspace(0, self.max_length-1.0, self.max_length)])).transpose(), (1, self.batch_size)),
                               name='A', borrow=True)
        self.s = T.nnet.sigmoid(self.W[0])  #sigmoid is used to contain parameter in [0, 1]
        self.b = T.log(1 - T.exp(-self.s)) / (self.max_length-1.0)
        self.factors = T.exp(self.s + (self.A * self.max_length / (self.length+1.0)) * self.b) + (1 - T.exp(self.s))
        self.factors_norm = self.factors / self.factors[0]
        self.output = T.sum(self.x * T.transpose(self.factors_norm).dimshuffle(0, 'x', 1), axis=2) / (self.length.dimshuffle(0, 'x') + 1.0)

        self.params = [self.W]
        self.lin = theano.shared(value=numpy.linspace(0, self.max_length - 1.0, self.max_length), name='lin', borrow=True)
        self.factors_30 = T.exp(self.s + self.A * self.b) + (1 - T.exp(self.s))

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
        self.s = T.nnet.sigmoid(self.W[0])
        self.b = T.log(1 - T.exp(-self.s)) / (self.max_length-1.0)
        self.factors = T.exp(self.s + (self.A * self.max_length / (self.length+1.0)) * self.b) + (1 - T.exp(self.s))
        self.factors_norm = self.factors / self.factors[0]
        return T.sum(self.x * T.transpose(self.factors_norm).dimshuffle(0, 'x', 1), axis=2) / (self.length.dimshuffle(0, 'x') + 1.0)


class exponentialTweetDotMeanLayer():
    """Layer calculating dot product of input with weights based on length, and normalizing.
       Weights are calculated through an exponential function, of which the coefficients
       are begin learned (intercept = 1.0)."""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, indices, length, max_length=30, n_out=1, batch_size=100, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.indices = indices #2D tensor
        self.length = length #1D tensor
        self.max_length = float(max_length)
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.batch_size = batch_size

        if W is None:
            W_values = numpy.asarray([-1.6], # a*exp(b*x) + 1 - a; b = ln(1 - 1/a)/(max_length-1); a = softplus(param) + 1
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.s = T.nnet.softplus(self.W[0]) + 1.0  #softplus is used to contain parameter in [1, inf]
        self.b = T.log(1 - 1/self.s) / (self.max_length-1.0)
        self.factors = self.s * T.exp(self.indices * self.b) + (1 - self.s)
        self.factors_norm = self.factors / self.factors[0]
        self.output = T.sum(self.x * T.transpose(self.factors_norm).dimshuffle(0, 'x', 1), axis=2) / \
                      (self.length + 1.0).dimshuffle(0, 'x')

        self.params = [self.W]
        self.A = theano.shared(value=numpy.tile((numpy.asarray([numpy.linspace(0, self.max_length-1.0, self.max_length)])).transpose(), (1, self.batch_size)),
                               name='A', borrow=True)
        self.lin = theano.shared(value=numpy.linspace(0, self.max_length - 1.0, self.max_length), name='lin', borrow=True)
        self.factors_30 = T.exp(self.s + self.A * self.b) + (1 - T.exp(self.s))

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
        self.s = T.nnet.softplus(self.W[0])+1.0  #softplus is used to contain parameter in [1, inf]
        self.b = T.log(1 - 1/self.s) / (self.max_length-1.0)
        self.factors = self.s * T.exp(self.indices * self.b) + (1 - self.s)
        self.factors_norm = self.factors / self.factors[0]
        self.output = T.sum(self.x * T.transpose(self.factors_norm).dimshuffle(0, 'x', 1), axis=2) / \
                      (self.length + 1.0).dimshuffle(0, 'x')


class interpolatingDotMeanLayer():
    """Layer calculating dot product of input with interpolated weights and taking the mean"""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, indices, length, max_length=30, n_out=1, batch_size=100, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.indices = indices #2D tensor
        self.length = length #1D tensor
        self.max_length = float(max_length)
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        init_W = [ 0.54457003,  0.72741562,  1.39331913,  1.12367916,  0.79878163,
        0.27706152,  0.3593896 ,  0.39622781,  0.27895978,  0.23260947,
        0.26763204,  0.27084899,  0.07067534,  0.13463201,  0.07948229,
        0.02779013,  0.12053657,  0.14807181,  0.24277158,  0.36964679,
        0.1601541 ,  0.37342793,  0.47257897,  0.39729786,  0.56589139,
        0.30535939,  0.10021771,  0.07151619,  0.12510002,  0.3112531 ,
        0.43562451,  0.05050614,  0.07199406,  0.50659907,  0.42588547]

        if W is None:
            W_values = numpy.asarray(
                    # self.numpy_rng.uniform(
                    #     low=0.5,
                    #     high=0.5,
                    #     size=(self.n_in)
                    # ),
                    init_W,
                    # numpy.linspace(1.0, 0.0, self.n_in),
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.indices_high = T.ceil(self.indices).astype('int8')
        self.indices_low = T.floor(self.indices).astype('int8')
        self.factors_high = self.W[self.indices_high]
        self.factors_low = self.W[self.indices_low]
        self.factors = (self.factors_high - self.factors_low) * (self.indices - self.indices_low) / \
                       (self.indices_high - self.indices_low + 1E-5) + self.factors_low
        self.output = T.sum(self.x * T.transpose(self.factors).dimshuffle(0, 'x', 1), axis=2) / \
                      (self.length + 1.0).dimshuffle(0, 'x')

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
        self.indices_high = T.ceil(self.indices).astype('int8')
        self.indices_low = T.floor(self.indices).astype('int8')
        self.factors_high = self.W[self.indices_high]
        self.factors_low = self.W[self.indices_low]
        self.factors = (self.factors_high - self.factors_low) * (self.indices - self.indices_low) / \
                       (self.indices_high - self.indices_low + 1E-5) + self.factors_low
        self.output = T.sum(self.x * T.transpose(self.factors).dimshuffle(0, 'x', 1), axis=2) / \
                      (self.length + 1.0).dimshuffle(0, 'x')


class linDotMeanLayer():
    """Layer calculating dot product of input with linear weights and taking the mean"""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, indices, length, max_length=30, n_out=1, batch_size=100, W=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.indices = indices #2D tensor
        self.length = length #1D tensor
        self.max_length = float(max_length)
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng

        init_W = [ 0.73289382,  0.61705852,  0.5028789 ,  0.4221231 ,  0.39812899,
        0.42901528,  0.59032136,  0.67545795,  0.69947743,  0.61530566,
        0.54543543,  0.49329555,  0.464315  ,  0.45519873,  0.45874992,
        0.45487615,  0.46938628,  0.47694537,  0.47277218,  0.48079902,
        0.491207  ,  0.49519542,  0.4995243 ,  0.49758726,  0.50004548,
        0.49860209,  0.49975339,  0.50008887,  0.50006837,  0.50001836,
        0.50004667,  0.49994707,  0.50002164,  0.50002164,  0.49999908]

        if W is None:
            W_values = numpy.asarray(
                    self.numpy_rng.uniform(
                        low=0.5,
                        high=0.5,
                        size=(self.n_in)
                    ),
                    # init_W,
                    # numpy.linspace(1.0, 0.0, self.n_in),
                    dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        self.factors = self.W[self.indices.astype('int8')]
        self.output = T.sum(self.x * T.transpose(self.factors).dimshuffle(0, 'x', 1), axis=2) / \
                      (self.length + 1.0).dimshuffle(0, 'x')

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
        self.factors = self.W[self.indices]
        self.output = T.sum(self.x * T.transpose(self.factors).dimshuffle(0, 'x', 1), axis=2) / \
                      (self.length + 1.0).dimshuffle(0, 'x')


class MLPLayer():
    """Layer calculating dot product of input with weights and taking the mean"""

    def __init__(self, numpy_rng, theano_rng, input, input_shape, n_out=1, batch_size=100, W=None, b=None):
        self.n_out = n_out
        self.n_in = input_shape[1]
        self.x = input #3D tensor
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.batch_size = batch_size
        self.input_shape = input_shape

        if W is None:
            W_values = numpy.asarray(
                self.numpy_rng.uniform(
                    low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
                    high=numpy.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_out)
                ),
                dtype=theano.config.floatX
                )
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W

        if b is None:
            b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            self.b = b

        self.output = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b).reshape((self.batch_size, self.input_shape[0]))

        self.params = [self.W, self.b]

    def save_me(self, filename=None):
        f = open(filename, 'wb')
        numpy.savez_compressed(f, self.W.get_value(borrow=True))
        numpy.savez_compressed(f, self.b.get_value(borrow=True))
        f.close()

    def load_me(self, filename=None):
        f = open(filename, 'rb')
        dictionary = numpy.load(f)
        self.W.set_value(dictionary['arr_0'])
        self.b.set_value(dictionary['arr_1'])
        f.close()

    def get_hidden_values(self, input, batch_size):
        return T.nnet.sigmoid(T.dot(self.x, self.W) + self.b).reshape((self.batch_size, self.input_shape[0]))